from copy import copy

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor

from embeddings.common.log import logger
from embeddings.common.paths import ModelPaths, ModelPathsCreator, PlotPaths
from embeddings.evaluation.inverse_problem import InverseProblem
from embeddings.models.vae.vae import VariationalAutoEncoder

from ._inverse_problem_solver import InverseProblemSolver

# Empirically determined
_LEARNING_RATES = {
    10: 1.5e-4,
    25: 5e-4,
    50: 1e-3,
    100: 1.8e-3,
    250: 2.5e-3,
    500: 4e-3,
    1_000: 5.5e-3,
    2_500: 7e-3,
    5_000: 1.5e-2,
    10_000: 3e-2,
    12_500: 4e-2,
}


class GenerativeModelSolver(InverseProblemSolver):
    MAX_STEPS = 10_000
    STOP_AFTER = 250

    def __init__(
        self,
        plot_loss: bool = False,
        log_info: bool = False,
        path_to_model: ModelPaths | None = None,
    ) -> None:
        self._plot_loss = plot_loss
        self._log_info = log_info
        self._load_generator(path=path_to_model)

    def _load_generator(self, path: ModelPaths | None) -> None:
        model_path = path if path else ModelPathsCreator.get_latest_vae_model()
        vae = VariationalAutoEncoder.load_from_checkpoint(checkpoint_path=model_path.checkpoint)
        self._device = vae.device
        self._generator = vae.decoder
        self._latent_dimension = vae.latent_dimension

    def _generate(self, z: Tensor) -> Tensor:
        x_rec = self._generator(z)
        return x_rec.view(15 * 32 * 32)

    def _target(self, A: Tensor, y: Tensor, z: Tensor) -> Tensor:  # noqa: N803
        loss = torch.norm(y - A @ self._generate(z)).pow(2)
        regularization = torch.norm(z).pow(2)
        return loss + regularization

    def solve(self, inverse_problem: InverseProblem) -> Tensor:
        z = torch.randn(self._latent_dimension).to(self._device)
        z.requires_grad = True

        a_on_device = inverse_problem.A.to(self._device)
        y_on_device = inverse_problem.y.to(self._device)

        num_measurements = len(inverse_problem.y)
        learning_rate = _LEARNING_RATES[num_measurements]

        optimizer = torch.optim.Adam(params=[z], lr=learning_rate)

        losses = []

        cur_best_z = z
        min_loss = float("inf")

        no_improvements = 0

        stopped_at = -1

        for iteration in range(self.MAX_STEPS):
            loss = self._target(A=a_on_device, y=y_on_device, z=z)
            loss.backward()
            optimizer.step()

            if loss.item() < min_loss:
                no_improvements = 0
                cur_best_z = copy(z)
                min_loss = loss.item()
            else:
                no_improvements += 1

            losses.append(np.log(loss.item()))

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
            plt.plot(losses)
            plt.savefig(PlotPaths.PLOTS / "loss.png")

        return self._generate(cur_best_z).cpu().detach()
