import numpy as np
import pywt
from sklearn.linear_model import Lasso
from torch import Tensor

from embeddings.common.gnfr_sector import NUM_GNFR_SECTORS
from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection
from embeddings.inverse_problems.inverse_problem import InverseProblem

from ._inverse_problem_solver import InverseProblemSolver

_WIDTH = TnoDatasetCollection.CROPPED_WIDTH
_HEIGHT = TnoDatasetCollection.CROPPED_HEIGHT
_DEPTH = NUM_GNFR_SECTORS

_UNSUPPORTED_DIMENSIONS_ERROR = "Dimension for sensing matrix is not supported."


class DwtLassoSolver(InverseProblemSolver):

    def solve(self, inverse_problem: InverseProblem) -> Tensor:
        A = inverse_problem.A.numpy()  # noqa: N806
        y = inverse_problem.y.numpy()

        lasso = Lasso(alpha=0.1, max_iter=10_000)

        A_DWT, coefficient_slices = self._transform_sensing_matrix(sensing_matrix=A)  # noqa: N806
        lasso.fit(A_DWT, y)
        c = lasso.coef_
        x = self._inverse_transform_emission_field(coefficients=c, coefficient_slices=coefficient_slices)

        return Tensor(x)

    def _transform_sensing_matrix(self, sensing_matrix: np.array) -> tuple[np.array, list[tuple]]:
        transformed_sensing_matrix = np.zeros(sensing_matrix.shape)
        coefficient_slices: list[tuple] = []
        for row in range(sensing_matrix.shape[0]):
            coefficients, coefficient_slices = self._transform_emission_field(field_as_vec=sensing_matrix[row, :])
            transformed_sensing_matrix[row, :] = coefficients
        return transformed_sensing_matrix, coefficient_slices

    def _transform_emission_field(self, field_as_vec: np.array) -> tuple[np.array, list[tuple]]:
        if len(field_as_vec) == _DEPTH * _HEIGHT * _WIDTH:
            return self._transform_emission_field_sector_wise(field_as_vec)
        if len(field_as_vec) == _WIDTH * _HEIGHT:
            return self._transform_emission_field_total(field_as_vec)
        raise AttributeError(_UNSUPPORTED_DIMENSIONS_ERROR)

    def _transform_emission_field_sector_wise(self, field_as_vec: np.array) -> tuple[np.array, list[tuple]]:
        field = field_as_vec.reshape((_DEPTH, _HEIGHT, _WIDTH))
        transformed_field = np.zeros(field.shape)
        coefficient_slices = []
        for sector in range(_DEPTH):
            wavelet_coefficients = pywt.wavedec2(field[sector, :, :], "haar", level=3)
            wavelet_coefficients, coefficient_slices = pywt.coeffs_to_array(wavelet_coefficients)
            transformed_field[sector, :, :] = wavelet_coefficients
        return transformed_field.reshape(_DEPTH * _HEIGHT * _WIDTH), coefficient_slices

    def _transform_emission_field_total(self, field_as_vec: np.array) -> tuple[np.array, list[tuple]]:
        field = field_as_vec.reshape((_HEIGHT, _WIDTH))
        wavelet_coefficients = pywt.wavedec2(field, "haar", level=3)
        transformed_field, coefficient_slices = pywt.coeffs_to_array(wavelet_coefficients)
        return transformed_field.reshape(_HEIGHT * _WIDTH), coefficient_slices

    def _inverse_transform_emission_field(
        self,
        coefficients: np.array,
        coefficient_slices: list[tuple],
    ) -> np.array:
        if len(coefficients) == _DEPTH * _HEIGHT * _WIDTH:
            return self._inverse_transform_emission_field_sector_wise(coefficients, coefficient_slices)
        if len(coefficients) == _WIDTH * _HEIGHT:
            return self._inverse_transform_emission_field_total(coefficients, coefficient_slices)
        raise AttributeError(_UNSUPPORTED_DIMENSIONS_ERROR)

    def _inverse_transform_emission_field_sector_wise(
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

    def _inverse_transform_emission_field_total(
        self,
        coefficients: np.array,
        coefficient_slices: list[tuple],
    ) -> np.array:
        coefficients_as_field = coefficients.reshape((_HEIGHT, _WIDTH))
        field_as_array = pywt.array_to_coeffs(
            coefficients_as_field,
            coefficient_slices,
            output_format="wavedec2",
        )
        field = pywt.waverec2(field_as_array, "haar")
        return field.reshape(_HEIGHT * _WIDTH)

    def _generate_wavelet_basis(self) -> np.array:
        """
        Alternative implementation of wavelet transform (only works for sector-wise transform).
        """
        sector_size = _WIDTH * _HEIGHT
        sector_basis = self._generate_wavelet_basis_for_sector(wavelet="haar", levels=3)
        wavelet_basis = np.zeros((_DEPTH * sector_size, _DEPTH * sector_size))
        for i in range(_DEPTH):
            wavelet_basis[i * sector_size : (i + 1) * sector_size, i * sector_size : (i + 1) * sector_size] = (
                sector_basis
            )
        return wavelet_basis

    def _generate_wavelet_basis_for_sector(self, wavelet: str, levels: int) -> np.array:
        sector = np.zeros((_WIDTH, _HEIGHT))
        coefficients = pywt.wavedec2(sector, wavelet, levels)
        basis = []
        for i in range(len(coefficients)):
            coefficients[i] = list(coefficients[i])
            n_filters = len(coefficients[i])
            for j in range(n_filters):
                for m in range(coefficients[i][j].shape[0]):
                    if coefficients[i][j].ndim == 1:
                        coefficients[i][j][m] = 1
                        temp_basis = pywt.waverec2(coefficients, wavelet)
                        basis.append(temp_basis)
                        coefficients[i][j][m] = 0
                    else:
                        for n in range(coefficients[i][j].shape[1]):
                            coefficients[i][j][m][n] = 1
                            temp_basis = pywt.waverec2(coefficients, wavelet)
                            basis.append(temp_basis)
                            coefficients[i][j][m][n] = 0
        return np.array(basis).reshape((_WIDTH * _HEIGHT, _WIDTH * _HEIGHT)).T