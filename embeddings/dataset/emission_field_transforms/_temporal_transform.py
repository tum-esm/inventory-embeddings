import polars as pl

from embeddings.common.gnfr_sector import NUM_GNFR_SECTORS, GnfrSector
from embeddings.common.paths import TnoPaths
from embeddings.dataset.city_emission_field import CityEmissionField

from ._emission_field_transforms import EmissionFieldTransform


class TimeProfiles:
    _HOUR_TIME_PROFILE = None

    @classmethod
    def get_hour_time_profile(cls) -> pl.DataFrame:
        if cls._HOUR_TIME_PROFILE is None:
            cls._HOUR_TIME_PROFILE = pl.read_csv(TnoPaths.HOUR_TIME_PROFILE, separator=";", skip_rows=6)
        return cls._HOUR_TIME_PROFILE


class HourTransform(EmissionFieldTransform):
    def __init__(self, hour: int) -> None:
        self._scaling_factors = [1 for _ in range(NUM_GNFR_SECTORS)]
        for row in TimeProfiles.get_hour_time_profile().iter_rows(named=True):
            if row[str(hour)]:
                sector = GnfrSector.from_str(row["TNO GNFR sectors Sept 2018"])
                self._scaling_factors[sector.to_index()] = row[str(hour)]
        print(self._scaling_factors)

    def __call__(self, emission_field: CityEmissionField) -> CityEmissionField:
        for sector, scaling_factor in enumerate(self._scaling_factors):
            emission_field.co2_ff_tensor[:, :, sector] *= scaling_factor
        return emission_field
