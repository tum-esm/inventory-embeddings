import numpy as np
from scipy import fftpack

from src.common.gnfr_sector import NUM_GNFR_SECTORS
from src.dataset.tno_dataset_collection import TnoDatasetCollection

_WIDTH = TnoDatasetCollection.CROPPED_WIDTH
_HEIGHT = TnoDatasetCollection.CROPPED_HEIGHT
_DEPTH = NUM_GNFR_SECTORS

_UNSUPPORTED_DIMENSIONS_ERROR = "Dimension for sensing matrix is not supported."


def dct2(array: np.array) -> np.array:
    return fftpack.dct(fftpack.dct(array.T, norm="ortho").T, norm="ortho")


def inverse_dct2(array: np.array) -> np.array:
    return fftpack.idct(fftpack.idct(array.T, norm="ortho").T, norm="ortho")


class DctTransform:
    @staticmethod
    def transform_sensing_matrix(sensing_matrix: np.array) -> np.array:
        transformed_sensing_matrix = np.zeros(sensing_matrix.shape)
        for row in range(sensing_matrix.shape[0]):
            transformed_row = DctTransform._transform_emission_field(sensing_matrix[row, :])
            transformed_sensing_matrix[row, :] = transformed_row
        return transformed_sensing_matrix

    @staticmethod
    def inverse_transform_field(transformed_field_as_vec: np.array) -> np.array:
        if len(transformed_field_as_vec) == _DEPTH * _HEIGHT * _WIDTH:
            return DctTransform._inverse_transform_emission_field_sector_wise(transformed_field_as_vec)
        if len(transformed_field_as_vec) == _WIDTH * _HEIGHT:
            return DctTransform._inverse_transform_emission_field_total(transformed_field_as_vec)
        raise AttributeError(_UNSUPPORTED_DIMENSIONS_ERROR)

    @staticmethod
    def _transform_emission_field(field_as_vec: np.array) -> np.array:
        if len(field_as_vec) == _DEPTH * _HEIGHT * _WIDTH:
            return DctTransform._transform_emission_field_sector_wise(field_as_vec)
        if len(field_as_vec) == _WIDTH * _HEIGHT:
            return DctTransform._transform_emission_field_total(field_as_vec)
        raise AttributeError(_UNSUPPORTED_DIMENSIONS_ERROR)

    @staticmethod
    def _transform_emission_field_sector_wise(field_as_vec: np.array) -> np.array:
        field = field_as_vec.reshape((_DEPTH, _HEIGHT, _WIDTH))
        transformed_field = np.zeros(field.shape)
        for sector in range(_DEPTH):
            sector_field = field[sector, :, :]
            transformed_field[sector, :, :] = dct2(sector_field)
        return transformed_field.reshape(_DEPTH * _HEIGHT * _WIDTH)

    @staticmethod
    def _transform_emission_field_total(field_as_vec: np.array) -> np.array:
        field = field_as_vec.reshape((_HEIGHT, _WIDTH))
        transformed_field = dct2(field)
        return transformed_field.reshape(_HEIGHT * _WIDTH)

    @staticmethod
    def _inverse_transform_emission_field_sector_wise(transformed_field_as_vec: np.array) -> np.array:
        transformed_field = transformed_field_as_vec.reshape((_DEPTH, _HEIGHT, _WIDTH))
        field = np.zeros(transformed_field.shape)
        for sector in range(_DEPTH):
            transformed_sector_field = transformed_field[sector, :, :]
            field[sector, :, :] = inverse_dct2(transformed_sector_field)
        return field.reshape(_DEPTH * _HEIGHT * _WIDTH)

    @staticmethod
    def _inverse_transform_emission_field_total(transformed_field_as_vec: np.array) -> np.array:
        transformed_field = transformed_field_as_vec.reshape((_HEIGHT, _WIDTH))
        field = inverse_dct2(transformed_field)
        return field.reshape(_HEIGHT * _WIDTH)
