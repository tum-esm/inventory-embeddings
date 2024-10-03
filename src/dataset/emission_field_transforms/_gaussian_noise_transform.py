import numpy as np

from src.dataset.city_emission_field import CityEmissionField

from . import EmissionFieldTransform


class GaussianNoiseTransform(EmissionFieldTransform):
    def __call__(self, emission_field: CityEmissionField) -> CityEmissionField:
        f = emission_field

        generator = np.random.default_rng()

        min_area_source = f.co2_ff_area_sources_field.min()
        area_sources_noise = generator.normal(0, 0.1, f.co2_ff_area_sources_field.shape)
        f.co2_ff_area_sources_field += area_sources_noise
        f.co2_ff_area_sources_field = np.clip(f.co2_ff_area_sources_field, a_min=min_area_source, a_max=None)

        min_point_source = f.co2_ff_point_sources_field.min()
        point_sources_noise = generator.normal(0, 0.1, f.co2_ff_point_sources_field.shape)
        f.co2_ff_point_sources_field *= point_sources_noise
        f.co2_ff_point_sources_field = np.clip(f.co2_ff_point_sources_field, a_min=min_point_source, a_max=None)

        return f
