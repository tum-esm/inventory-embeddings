import numpy as np

from src.common.gnfr_sector import NUM_GNFR_SECTORS
from src.dataset.city_emission_field import CityEmissionField

from ._emission_field_transform import EmissionFieldTransform


class RandomSparseEmittersTransform(EmissionFieldTransform):
    def __init__(self, lam: int = 20) -> None:
        self._lam = lam
        self._rng = np.random.default_rng()

    def __call__(self, emission_field: CityEmissionField) -> CityEmissionField:
        width = emission_field.width
        height = emission_field.height

        low = 0.5
        high = 1.0

        s = int(self._rng.poisson(lam=self._lam, size=1))
        random_sectors = self._rng.integers(low=0, high=NUM_GNFR_SECTORS, size=s)
        random_x_index = self._rng.integers(low=0, high=width, size=s)
        random_y_index = self._rng.integers(low=0, high=height, size=s)

        random_scaling_factors = low + (high - low) * self._rng.random(size=s)
        for sector, scaling_factor, x, y in zip(random_sectors, random_scaling_factors, random_x_index, random_y_index):
            max_value_in_sector = emission_field.co2_ff_field[sector, :, :].max()
            emission_field.co2_ff_field[sector, y, x] = scaling_factor * max_value_in_sector
        return emission_field
