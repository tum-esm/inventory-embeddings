import numpy as np
import pywt
from sklearn.linear_model import Lasso
from torch import Tensor

from embeddings.common.gnfr_sector import NUM_GNFR_SECTORS
from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection
from embeddings.evaluation.inverse_problem import InverseProblem

from ._inverse_problem_solver import InverseProblemSolver

_WIDTH = TnoDatasetCollection.CROPPED_WIDTH
_HEIGHT = TnoDatasetCollection.CROPPED_HEIGHT
_DEPTH = NUM_GNFR_SECTORS


class DwtLassoSolver(InverseProblemSolver):
    def solve(self, inverse_problem: InverseProblem) -> Tensor:
        A = inverse_problem.A.numpy()  # noqa: N806
        y = inverse_problem.y.numpy()

        lasso = Lasso(alpha=0.1, max_iter=10_000)

        A_DWT, coefficient_slices = self._wavelet_transform_sensing_matrix(sensing_matrix=A)  # noqa: N806
        lasso.fit(A_DWT, y)
        c = lasso.coef_
        x = self._reverse_wavelet_transform_emission_field(c, coefficient_slices)

        return Tensor(x)

    def _wavelet_transform_sensing_matrix(self, sensing_matrix: np.array) -> tuple[np.array, list[tuple]]:
        num_rows = sensing_matrix.shape[0]
        transformed_sensing_matrix = np.zeros(sensing_matrix.shape)
        coefficient_slices: list[tuple] = []
        for row in range(num_rows):
            field = sensing_matrix[row, :].reshape((_DEPTH, _HEIGHT, _WIDTH))
            coefficients, coefficient_slices = self._wavelet_transform_emission_field(field=field)
            transformed_sensing_matrix[row, :] = coefficients.reshape(_DEPTH * _HEIGHT * _WIDTH)
        return transformed_sensing_matrix, coefficient_slices

    def _wavelet_transform_emission_field(self, field: np.array) -> tuple[np.array, list[tuple]]:
        transformed_field = np.zeros(field.shape)
        coefficient_slices = []
        for sector in range(_DEPTH):
            wavelet_coefficients = pywt.wavedec2(field[sector, :, :], "haar", level=3)
            wavelet_coefficients, coefficient_slices = pywt.coeffs_to_array(wavelet_coefficients)
            transformed_field[sector, :, :] = wavelet_coefficients
        return transformed_field, coefficient_slices

    def _reverse_wavelet_transform_emission_field(
        self,
        coefficients: np.array,
        coefficient_slices: list[tuple],
    ) -> np.array:
        coefficients_as_field = coefficients.reshape((_DEPTH, _HEIGHT, _WIDTH))
        field = np.zeros(coefficients_as_field.shape)
        for channel in range(_DEPTH):
            channel_as_array = pywt.array_to_coeffs(
                coefficients_as_field[channel, :, :],
                coefficient_slices,
                output_format="wavedec2",
            )
            field[channel, :, :] = pywt.waverec2(channel_as_array, "haar")
        return field.reshape(_DEPTH * _HEIGHT * _WIDTH)
