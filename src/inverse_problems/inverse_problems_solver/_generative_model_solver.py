from copy import copy
from typing import Any, Self

import torch
from matplotlib import pyplot as plt
from torch import Tensor

from src.common.gnfr_sector import NUM_GNFR_SECTORS
from src.common.log import logger
from src.common.paths import ModelPaths, ModelPathsCreator, PlotPaths
from src.dataset.tno_dataset_collection import TnoDatasetCollection
from src.inverse_problems.inverse_problem import InverseProblem
from src.models.vae.vae import VariationalAutoEncoder

from ._inverse_problem_solver import InverseProblemSolver
from ._learning_rates_generative_model_solver import LEARNING_RATES

_WIDTH = TnoDatasetCollection.CROPPED_WIDTH
_HEIGHT = TnoDatasetCollection.CROPPED_HEIGHT
_DEPTH = NUM_GNFR_SECTORS

_UNSUPPORTED_DIMENSIONS_ERROR = "Dimension for sensing matrix is not supported."


class GenerativeModelSolver(InverseProblemSolver):
    MAX_STEPS = 10_000
    STOP_AFTER = 250
    IMPROVEMENT_TOLERANCE = 1e-5

    def __init__(
        self,
        *,
        plot_loss: bool = False,
        log_info: bool = False,
        path_to_model: ModelPaths | None = None,
    ) -> None:
        self._sector_wise_reconstruction = False
        self._plot_loss = plot_loss
        self._log_info = log_info
        self._load_generator(path=path_to_model)

    @classmethod
    def from_vae_model_name(
        cls,
        name: str,
        *,
        plot_loss: bool = False,
        log_info: bool = False,
    ) -> Self:
        return cls(
            path_to_model=ModelPathsCreator.get_vae_model(name),
            plot_loss=plot_loss,
            log_info=log_info,
        )

    def _load_generator(self, path: ModelPaths | None) -> None:
        model_path = path if path else ModelPathsCreator.get_latest_vae_model()
        vae = VariationalAutoEncoder.load_from_checkpoint(checkpoint_path=model_path.checkpoint)
        vae.eval()
        self._device = vae.device
        self._generator = vae.decoder
        self._latent_dimension = vae.latent_dimension

    def _generate(self, z: Tensor) -> Tensor:
        x_rec = self._generator(z)
        if self._sector_wise_reconstruction:
            x_rec = x_rec.view(_DEPTH * _HEIGHT * _WIDTH)
        else:
            x_rec = x_rec.sum(dim=1)
            x_rec = x_rec.view(_HEIGHT * _WIDTH)
        return x_rec

    def _target(self, A: Tensor, y: Tensor, z: Tensor, gamma: float) -> Tensor:  # noqa: N803
        loss = torch.norm(y - A @ self._generate(z), p=2).pow(2)
        regularization = torch.norm(z, p=2).pow(2)
        return loss + gamma * regularization

    def _determine_if_reconstruction_is_sector_wise(self, inverse_problem: InverseProblem) -> None:
        first_row_of_sensing_matrix = inverse_problem.A[0, :]
        if len(first_row_of_sensing_matrix) == _DEPTH * _HEIGHT * _WIDTH:
            self._sector_wise_reconstruction = True
        elif len(first_row_of_sensing_matrix) == _HEIGHT * _WIDTH:
            self._sector_wise_reconstruction = False
        else:
            raise AttributeError(_UNSUPPORTED_DIMENSIONS_ERROR)

    def _plot_losses(self, losses: list[float]) -> None:
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.plot(losses)
        plt.xlim(0, len(losses) - 1)
        plt.savefig(PlotPaths.PLOTS / "loss.png")

    def solve(self, inverse_problem: InverseProblem, **settings: dict[str, Any]) -> Tensor:
        gamma = settings.pop("gamma", 0.0)
        learning_rate = settings.pop("learning_rate", None)

        if learning_rate is None:
            learning_rate = LEARNING_RATES[self._latent_dimension][inverse_problem.A.shape[0]]

        if not isinstance(gamma, float):
            raise TypeError
        if not isinstance(learning_rate, float):
            raise TypeError

        self._determine_if_reconstruction_is_sector_wise(inverse_problem)

        a_on_device = inverse_problem.A.to(self._device)
        y_on_device = inverse_problem.y.to(self._device)

        z = torch.randn(self._latent_dimension).to(self._device)
        z.requires_grad = True

        optimizer = torch.optim.Adam(params=[z], lr=learning_rate)

        losses = []

        cur_best_z = z
        min_loss = float("inf")

        no_improvements = 0

        stopped_at = -1

        for iteration in range(self.MAX_STEPS):
            loss = self._target(A=a_on_device, y=y_on_device, z=z, gamma=gamma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(z, max_norm=0.5)
            optimizer.step()

            percentage_improvement = (min_loss - loss.item()) / min_loss

            if percentage_improvement < self.IMPROVEMENT_TOLERANCE:
                no_improvements += 1
            else:
                no_improvements = 0

            if loss.item() < min_loss:
                cur_best_z = copy(z)
                min_loss = loss.item()

            losses.append(loss.item())

            if no_improvements == self.STOP_AFTER:
                stopped_at = iteration
                break

        if self._log_info:
            if stopped_at >= 0:
                logger.info(f"Optimization stopped at iteration {stopped_at} with minimum loss {min_loss}!")
            else:
                logger.warning(
                    f"Optimization stopped with minimum loss {min_loss}. "
                    f"However, the loss was still decreasing. "
                    f"Consider increasing the learning rate!",
                )

        if self._plot_loss:
            self._plot_losses(losses)

        return self._generate(cur_best_z).cpu().detach()
