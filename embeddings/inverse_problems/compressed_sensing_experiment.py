import numpy as np
import torch
from torch import Tensor

from embeddings.common.gnfr_sector import NUM_GNFR_SECTORS
from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection
from embeddings.inverse_problems.inverse_problem import InverseProblem
from embeddings.inverse_problems.inverse_problems_solver import InverseProblemSolver

_EMISSION_FIELD_WIDTH = TnoDatasetCollection.CROPPED_WIDTH
_EMISSION_FIELD_HEIGHT = TnoDatasetCollection.CROPPED_HEIGHT
_EMISSION_FIELD_DEPTH = NUM_GNFR_SECTORS
_EMISSION_FIELD_SIZE = _EMISSION_FIELD_WIDTH * _EMISSION_FIELD_HEIGHT * _EMISSION_FIELD_DEPTH
_EMISSION_FIELD_SPATIAL_SIZE = _EMISSION_FIELD_WIDTH * _EMISSION_FIELD_HEIGHT


def generate_random_inverse_problem(
    x: Tensor,
    num_measurements: int,
    signal_to_noise_ratio: int = 0,
) -> InverseProblem:
    sensing_matrix = torch.randn((num_measurements, _EMISSION_FIELD_SIZE))
    x_vectorized = _vectorize(x=x)

    measurements = sensing_matrix @ x_vectorized

    if signal_to_noise_ratio:
        measurements_np = np.array(measurements)
        signal_power = np.mean(measurements_np**2)
        noise_power = signal_power / signal_to_noise_ratio
        noise_std_dev = np.sqrt(noise_power)
        noise = np.random.default_rng().normal(0, noise_std_dev, num_measurements)
        measurements = Tensor(measurements_np + noise)

    return InverseProblem(A=sensing_matrix, y=measurements)


def generate_random_spatial_inverse_problem(
    x: Tensor,
    num_measurements: int,
    signal_to_noise_ratio: int = 0,
) -> InverseProblem:
    spatial_sensing_matrix = torch.randn((num_measurements, _EMISSION_FIELD_SPATIAL_SIZE))
    sensing_matrix = torch.concatenate(tuple(spatial_sensing_matrix for _ in range(_EMISSION_FIELD_DEPTH)), 1)
    x_vectorized = _vectorize(x=x)

    measurements = sensing_matrix @ x_vectorized

    if signal_to_noise_ratio:
        measurements_np = np.array(measurements)
        signal_power = np.mean(measurements_np**2)
        noise_power = signal_power / signal_to_noise_ratio
        noise_std_dev = np.sqrt(noise_power)
        noise = np.random.default_rng().normal(0, noise_std_dev, num_measurements)
        measurements = Tensor(measurements_np + noise)

    return InverseProblem(A=sensing_matrix, y=measurements)


def solve_inverse_problem(inverse_problem: InverseProblem, solver: InverseProblemSolver) -> Tensor:
    x_rec_vectorized = solver.solve(inverse_problem=inverse_problem)
    return _un_vectorize(x_rec_vectorized)


def _vectorize(x: Tensor) -> Tensor:
    return x.view(_EMISSION_FIELD_SIZE)


def _un_vectorize(x: Tensor) -> Tensor:
    return x.view(_EMISSION_FIELD_DEPTH, _EMISSION_FIELD_HEIGHT, _EMISSION_FIELD_WIDTH)
