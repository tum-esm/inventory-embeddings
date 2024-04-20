from enum import Enum

import numpy as np
import polars as pl

from embeddings.common.gnfr_sector import NUM_GNFR_SECTORS, GnfrSector
from embeddings.common.paths import TnoPaths
from embeddings.dataset.city_emission_field import CityEmissionField

from ._emission_field_transform import EmissionFieldTransform


class Weekday(Enum):
    Mon = " mon "
    Tue = " tue "
    Wed = " wed "
    Thu = " thu "
    Fri = " fri "
    Sat = " sat "
    Sun = " sun "


class Month(Enum):
    Jan = " jan "
    Feb = " feb  "
    Mar = " mar "
    Apr = " apr "
    May = " may "
    Jun = " jun "
    Jul = " jul "
    Aug = " aug  "
    Sep = " sep "
    Oct = " oct "
    Nov = " nov  "
    Dec = " dec"


class TimeProfiles:
    _HOUR_TIME_PROFILE = None
    _DAY_TIME_PROFILE = None
    _MONTH_TIME_PROFILE = None

    @classmethod
    def get_hour_time_profile(cls) -> pl.DataFrame:
        if cls._HOUR_TIME_PROFILE is None:
            cls._HOUR_TIME_PROFILE = pl.read_csv(TnoPaths.HOUR_TIME_PROFILE, separator=";", skip_rows=6)
        return cls._HOUR_TIME_PROFILE

    @classmethod
    def get_day_time_profile(cls) -> pl.DataFrame:
        if cls._DAY_TIME_PROFILE is None:
            cls._DAY_TIME_PROFILE = pl.read_csv(TnoPaths.DAY_TIME_PROFILE, separator=";", skip_rows=6)
        return cls._DAY_TIME_PROFILE

    @classmethod
    def get_month_time_profile(cls) -> pl.DataFrame:
        if cls._MONTH_TIME_PROFILE is None:
            cls._MONTH_TIME_PROFILE = pl.read_csv(TnoPaths.MONTH_TIME_PROFILE, separator=";", skip_rows=6)
        return cls._MONTH_TIME_PROFILE


class HourTransform(EmissionFieldTransform):
    def __init__(self, hour: int) -> None:
        self._scaling_factors = np.ones((NUM_GNFR_SECTORS, 1, 1))
        for row in TimeProfiles.get_hour_time_profile().iter_rows(named=True):
            factor = row[str(hour)]
            if factor:
                sector = GnfrSector.from_str(row["TNO GNFR sectors Sept 2018"])
                self._scaling_factors[sector.to_index(), 0, 0] = factor

    def __call__(self, emission_field: CityEmissionField) -> CityEmissionField:
        emission_field.co2_ff_field *= self._scaling_factors
        return emission_field


class DayTransform(EmissionFieldTransform):
    def __init__(self, week_day: Weekday) -> None:
        self._scaling_factors = np.ones((NUM_GNFR_SECTORS, 1, 1))
        for row in TimeProfiles.get_day_time_profile().iter_rows(named=True):
            factor = row[week_day.value]
            if factor:
                sector = GnfrSector.from_str(row["TNO GNFR sectors Sept 2018"])
                self._scaling_factors[sector.to_index(), 0, 0] = factor

    def __call__(self, emission_field: CityEmissionField) -> CityEmissionField:
        emission_field.co2_ff_field *= self._scaling_factors
        return emission_field


class MonthTransform(EmissionFieldTransform):
    def __init__(self, month: Month) -> None:
        self._scaling_factors = np.ones((NUM_GNFR_SECTORS, 1, 1))
        for row in TimeProfiles.get_month_time_profile().iter_rows(named=True):
            factor = row[month.value]
            if factor:
                sector = GnfrSector.from_str(row["TNO GNFR sectors Sept 2018"])
                self._scaling_factors[sector.to_index(), 0, 0] = factor

    def __call__(self, emission_field: CityEmissionField) -> CityEmissionField:
        emission_field.co2_ff_field *= self._scaling_factors
        return emission_field
