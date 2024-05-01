import torch
from torch import Tensor

from embeddings.common.gnfr_sector import NUM_GNFR_SECTORS
from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection
from embeddings.evaluation.inverse_problems_solver import InverseProblemSolver


class CompressedSensingExperiment:
    def __init__(self, num_measurements: int) -> None:
        self._num_measurements = num_measurements
        self._emission_field_width = TnoDatasetCollection.CROPPED_WIDTH
        self._emission_field_height = TnoDatasetCollection.CROPPED_HEIGHT
        self._emission_field_depth = NUM_GNFR_SECTORS
        self._emission_field_size = self._emission_field_depth * self._emission_field_width * self._emission_field_width

    def _generate_random_forward_model(self) -> Tensor:
        return torch.randn((self._num_measurements, self._emission_field_size))

    def _vectorize(self, x: Tensor) -> Tensor:
        return x.view(self._emission_field_size)

    def _un_vectorize(self, x: Tensor) -> Tensor:
        return x.view(self._emission_field_depth, self._emission_field_height, self._emission_field_height)

    def solve_random_inverse_problem(self, x: Tensor, solver: InverseProblemSolver) -> Tensor:
        A = self._generate_random_forward_model()  # noqa: N806
        x_vectorized = self._vectorize(x)
        y = A @ x_vectorized
        x_rec_vectorized = solver.solve(y=y, A=A)
        return self._un_vectorize(x_rec_vectorized)
