import numpy as np
from scipy import fftpack
from sklearn.linear_model import Lasso
from torch import Tensor

from embeddings.common.gnfr_sector import NUM_GNFR_SECTORS
from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection
from embeddings.inverse_problems.inverse_problem import InverseProblem

from ._inverse_problem_solver import InverseProblemSolver

_WIDTH = TnoDatasetCollection.CROPPED_WIDTH
_HEIGHT = TnoDatasetCollection.CROPPED_HEIGHT
_DEPTH = NUM_GNFR_SECTORS


def dct2(array: np.array) -> np.array:
    return fftpack.dct(fftpack.dct(array.T, norm="ortho").T, norm="ortho")


def inverse_dct2(array: np.array) -> np.array:
    return fftpack.idct(fftpack.idct(array.T, norm="ortho").T, norm="ortho")


class DctLassoSolver(InverseProblemSolver):
    def solve(self, inverse_problem: InverseProblem) -> Tensor:
        lasso = Lasso(alpha=0.1, max_iter=10_000)

        A = inverse_problem.A.numpy()  # noqa: N806
        y = inverse_problem.y.numpy()

        A_cosine_basis = self._transform_sensing_matrix(sensing_matrix=A)  # noqa: N806

        lasso.fit(A_cosine_basis, y)

        c = lasso.coef_

        x = self._inverse_transform_emission_field(transformed_field_as_vec=c)

        return Tensor(x)

    def _transform_sensing_matrix(self, sensing_matrix: np.array) -> np.array:
        transformed_sensing_matrix = np.zeros(sensing_matrix.shape)
        for row in range(sensing_matrix.shape[0]):
            transformed_row = self._transform_emission_field(sensing_matrix[row, :])
            transformed_sensing_matrix[row, :] = transformed_row
        return transformed_sensing_matrix

    def _transform_emission_field(self, field_as_vec: np.array) -> np.array:
        field = field_as_vec.reshape((_DEPTH, _HEIGHT, _WIDTH))
        transformed_field = np.zeros(field.shape)
        for sector in range(_DEPTH):
            sector_field = field[sector, :, :]
            transformed_field[sector, :, :] = dct2(sector_field)
        return transformed_field.reshape(_DEPTH * _HEIGHT * _WIDTH)

    def _inverse_transform_emission_field(self, transformed_field_as_vec: np.array) -> np.array:
        transformed_field = transformed_field_as_vec.reshape((_DEPTH, _HEIGHT, _WIDTH))
        field = np.zeros(transformed_field.shape)
        for sector in range(_DEPTH):
            transformed_sector_field = transformed_field[sector, :, :]
            field[sector, :, :] = inverse_dct2(transformed_sector_field)
        return field.reshape(_DEPTH * _HEIGHT * _WIDTH)
