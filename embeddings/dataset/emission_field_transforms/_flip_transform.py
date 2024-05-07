import random

import numpy as np

from embeddings.dataset.city_emission_field import CityEmissionField

from ._emission_field_transform import EmissionFieldTransform


class RandomHorizontalFlipTransform(EmissionFieldTransform):
    def __init__(self, p: float = 0.5) -> None:
        self._p = p

    def __call__(self, emission_field: CityEmissionField) -> CityEmissionField:
        number = random.random()
        if number <= self._p:
            emission_field.co2_ff_field_per_sector = np.flip(emission_field.co2_ff_field_per_sector, axis=2)
        return emission_field


class RandomVerticalFlipTransform(EmissionFieldTransform):
    def __init__(self, p: float = 0.5) -> None:
        self._p = p

    def __call__(self, emission_field: CityEmissionField) -> CityEmissionField:
        number = random.random()
        if number <= self._p:
            emission_field.co2_ff_field_per_sector = np.flip(emission_field.co2_ff_field_per_sector, axis=1)
        return emission_field
