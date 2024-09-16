import random

import numpy as np

from src.dataset.city_emission_field import CityEmissionField

from ._emission_field_transform import EmissionFieldTransform


class RandomRotationTransform(EmissionFieldTransform):
    def __init__(self, p: float = 0.5) -> None:
        self._p = p

    def __call__(self, emission_field: CityEmissionField) -> CityEmissionField:
        number = random.random()
        if number <= self._p:
            emission_field.co2_ff_field = np.rot90(emission_field.co2_ff_field, k=1, axes=(1, 2))
            emission_field.co2_ff_field_point_sources = np.rot90(
                emission_field.co2_ff_field_point_sources,
                k=1,
                axes=(1, 2),
            )
        return emission_field
