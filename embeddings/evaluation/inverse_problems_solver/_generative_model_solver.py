import torch
from torch import Tensor

from embeddings.common.paths import ModelPaths
from embeddings.models.vae.vae import VariationalAutoEncoder

from ._inverse_problem_solver import InverseProblemSolver


class GenerativeModelSolver(InverseProblemSolver):
    def __init__(self) -> None:
        self._load_generator()

    def _load_generator(self) -> None:
        first_check_point = next(ModelPaths.VAE_LATEST_CHECKPOINTS.iterdir())
        vae = VariationalAutoEncoder.load_from_checkpoint(checkpoint_path=first_check_point)
        self._generator = vae.decoder

    def solve(self, A: Tensor, y: Tensor) -> Tensor:  # noqa: ARG002, N803
        return torch.randn(15 * 32 * 32)
