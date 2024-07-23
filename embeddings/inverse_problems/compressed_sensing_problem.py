from typing import Self

import numpy as np
import torch
from torch import Tensor

from embeddings.common.gnfr_sector import NUM_GNFR_SECTORS
from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection
from embeddings.inverse_problems.inverse_problem import InverseProblem
from embeddings.inverse_problems.inverse_problems_solver import InverseProblemSolver

_DIMENSION_MISMATCH_ERROR = "Dimensions are emission field are not as expected!"

_EMISSION_FIELD_WIDTH = TnoDatasetCollection.CROPPED_WIDTH
_EMISSION_FIELD_HEIGHT = TnoDatasetCollection.CROPPED_HEIGHT
_EMISSION_FIELD_DEPTH = NUM_GNFR_SECTORS

class CompressedSensingProblem:
    def __init__(self, inverse_problem: InverseProblem) -> None:
        self._inverse_problem = inverse_problem

    def solve(self, solver: InverseProblemSolver) -> Tensor:
        x_rec_vectorized = solver.solve(inverse_problem=self._inverse_problem)
        return self._un_vectorize(x_rec_vectorized)

    @classmethod
    def _vectorize(cls, x: Tensor) -> Tensor: ...

    @classmethod
    def _un_vectorize(cls, x: Tensor) -> Tensor: ...

    @classmethod
    def _compute_measurement(cls, x: Tensor, sensing_matrix: Tensor, snr: int | None) -> Tensor:
        measurements = sensing_matrix @ cls._vectorize(x=x)

        if snr:
            measurements = cls._add_noise(snr=snr, measurements=measurements)

        return measurements

    @classmethod
    def _add_noise(cls, snr: int, measurements: Tensor) -> Tensor:
        num_measurements = len(measurements)
        measurements_np = np.array(measurements)
        signal_power = np.mean(measurements_np**2)
        noise_power = signal_power / snr
        noise_std_dev = np.sqrt(noise_power)
        noise = np.random.default_rng().normal(0, noise_std_dev, num_measurements)
        return Tensor(measurements_np + noise)


class SectorWiseCompressedSensingProblem(CompressedSensingProblem):
    """
    Compressed sensing experiments for reconstruction of emission fields per sector.
    Emission fields have dimension: number of sectors x emission field height x emission field width
    """
    _EMISSION_FIELD_SIZE = _EMISSION_FIELD_DEPTH * _EMISSION_FIELD_HEIGHT * _EMISSION_FIELD_WIDTH

    @classmethod
    def generate_random_sector_wise_measurements(
        cls,
        x: Tensor,
        num_measurements: int,
        snr: int | None = None,
    ) -> Self:
        """
        Sensing matrix is random Gaussian.
        This means that emissions of each sector can individually be sensed.
        While this is physically not possible in the real world,
        this type of problem allows evaluating the capabilities of different solvers.
        """
        cls._verify_dimensions_of_emission_field(x)

        sensing_matrix = torch.randn((num_measurements, cls._EMISSION_FIELD_SIZE))

        measurements = cls._compute_measurement(x=x, sensing_matrix=sensing_matrix, snr=snr)
        return cls(inverse_problem=InverseProblem(A=sensing_matrix, y=measurements))


    @classmethod
    def generate_random_total_emission_measurements(
        cls,
        x: Tensor,
        num_measurements: int,
        snr: int | None = None,
    ) -> Self:
        """
        Sensing matrix is random Gaussian with dimension:
            num_measurements x (emission field height * emission field width).
        Sectors are not individually sensed, but instead the total emissions are sensed.
        """
        cls._verify_dimensions_of_emission_field(x)

        spatial_sensing_matrix = torch.randn((num_measurements, _EMISSION_FIELD_WIDTH * _EMISSION_FIELD_HEIGHT))
        sensing_matrix = torch.concatenate(tuple(spatial_sensing_matrix for _ in range(_EMISSION_FIELD_DEPTH)), 1)

        measurements = cls._compute_measurement(x=x, sensing_matrix=sensing_matrix, snr=snr)
        return cls(inverse_problem=InverseProblem(A=sensing_matrix, y=measurements))

    @classmethod
    def generate_gaussian_plume_total_emission_measurements(
        cls,
        x: Tensor,
        num_measurements: int,
        snr: int | None = None,
    ) -> Self:
        """
        Sensing matrix is derived from a footprint randomly generated using a Gaussian plume model:
            num_measurements x (emission field height * emission field width).
        Sectors are not individually sensed, but instead the total emissions are sensed.
        """
        raise NotImplementedError

    @classmethod
    def _verify_dimensions_of_emission_field(cls, x: Tensor) -> None:
        if x.shape != (_EMISSION_FIELD_DEPTH, _EMISSION_FIELD_HEIGHT, _EMISSION_FIELD_WIDTH):
            raise ValueError(_DIMENSION_MISMATCH_ERROR)

    @classmethod
    def _vectorize(cls, x: Tensor) -> Tensor:
        return x.view(cls._EMISSION_FIELD_SIZE)

    @classmethod
    def _un_vectorize(cls, x: Tensor) -> Tensor:
        return x.view(_EMISSION_FIELD_DEPTH, _EMISSION_FIELD_HEIGHT, _EMISSION_FIELD_WIDTH)


class TotalEmissionsCompressedSensingExperiment(CompressedSensingProblem):
    """
    Compressed sensing experiment for reconstruction of total emissions independent of sector.
    Emission fields have dimension: emission field height x emission field width
    """
    _EMISSION_FIELD_SIZE = _EMISSION_FIELD_HEIGHT * _EMISSION_FIELD_WIDTH

    @classmethod
    def generate_random_measurements(
        cls,
        x: Tensor,
        num_measurements: int,
        snr: int | None = None,
    ) -> Self:
        """
        Sensing matrix is random Gaussian with dimension:
            num_measurements x (emission field height * emission field width).
        """
        cls._verify_dimensions_of_emission_field(x)

        sensing_matrix = torch.randn((num_measurements, cls._EMISSION_FIELD_SIZE))

        measurements = cls._compute_measurement(x=x, sensing_matrix=sensing_matrix, snr=snr)
        return cls(inverse_problem=InverseProblem(A=sensing_matrix, y=measurements))

    @classmethod
    def generate_gaussian_plume_measurements(
        cls,
        x: Tensor,
        num_measurements: int,
        snr: int | None = None,
    ) -> Self:
        raise NotImplementedError

    @classmethod
    def _verify_dimensions_of_emission_field(cls, x: Tensor) -> None:
        if x.shape != (_EMISSION_FIELD_HEIGHT, _EMISSION_FIELD_WIDTH):
            raise ValueError(_DIMENSION_MISMATCH_ERROR)

    @classmethod
    def _vectorize(cls, x: Tensor) -> Tensor:
        return x.view(cls._EMISSION_FIELD_SIZE)

    @classmethod
    def _un_vectorize(cls, x: Tensor) -> Tensor:
        return x.view(_EMISSION_FIELD_HEIGHT, _EMISSION_FIELD_WIDTH)
