import torch
from torch import Tensor

from embeddings.common.gnfr_sector import NUM_GNFR_SECTORS
from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection
from embeddings.evaluation.inverse_problem import InverseProblem
from embeddings.evaluation.inverse_problems_solver import InverseProblemSolver

_EMISSION_FIELD_WIDTH = TnoDatasetCollection.CROPPED_WIDTH
_EMISSION_FIELD_HEIGHT = TnoDatasetCollection.CROPPED_HEIGHT
_EMISSION_FIELD_DEPTH = NUM_GNFR_SECTORS
_EMISSION_FIELD_SIZE = _EMISSION_FIELD_WIDTH * _EMISSION_FIELD_HEIGHT * _EMISSION_FIELD_DEPTH


def generate_random_inverse_problem(x: Tensor, num_measurements: int) -> InverseProblem:
    sensing_matrix = torch.randn((num_measurements, _EMISSION_FIELD_SIZE))
    x_vectorized = _vectorize(x=x)
    # TODO(must1d): add noise for measurements  # noqa: FIX002, TD003
    measurements = sensing_matrix @ x_vectorized
    return InverseProblem(A=sensing_matrix, y=measurements)


def solve_inverse_problem(inverse_problem: InverseProblem, solver: InverseProblemSolver) -> Tensor:
    x_rec_vectorized = solver.solve(inverse_problem=inverse_problem)
    return _un_vectorize(x_rec_vectorized)


def _vectorize(x: Tensor) -> Tensor:
    return x.view(_EMISSION_FIELD_SIZE)


def _un_vectorize(x: Tensor) -> Tensor:
    return x.view(_EMISSION_FIELD_DEPTH, _EMISSION_FIELD_HEIGHT, _EMISSION_FIELD_WIDTH)
