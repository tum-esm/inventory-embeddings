import numpy as np

from src.common.gnfr_sector import NUM_GNFR_SECTORS
from src.dataset.city_emission_field import CityEmissionField

from . import EmissionFieldTransform


class SectorUncertaintyTransform(EmissionFieldTransform):
    def __call__(self, emission_field: CityEmissionField) -> CityEmissionField:
        generator = np.random.default_rng()
        scaling_factor = generator.uniform(0.5, 1.5, (NUM_GNFR_SECTORS, 1, 1))
        emission_field.co2_ff_area_sources_field *= scaling_factor
        emission_field.co2_ff_point_sources_field *= scaling_factor
        return emission_field
