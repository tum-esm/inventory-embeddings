from copy import deepcopy
from operator import attrgetter
from pathlib import Path
from typing import Self

import numpy as np
import polars as pl
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from tqdm import tqdm

from embeddings.common.log import logger
from embeddings.dataset.city_emission_field import CityEmissionField
from embeddings.dataset.emission_field_transforms import (
    DayTransform,
    EmissionFieldTransform,
    HourTransform,
    Month,
    MonthTransform,
    Weekday,
)


class TnoDataset(Dataset[Tensor]):
    def __init__(self, city_emission_fields: list[CityEmissionField]) -> None:
        self.city_emission_fields = city_emission_fields

        self._initialize_transforms()
        self._sampling_transforms: list[EmissionFieldTransform] = []

        self._temporal_transforms_enabled = True

    @classmethod
    def from_csv(cls, path: Path) -> Self:
        logger.info(f"Loading TNO data from '{path}'")
        tno_data = pl.read_csv(path, separator=";")
        city_emission_fields = cls._load_data(tno_data)
        city_emission_fields.sort(key=attrgetter("city_name"))
        city_emission_fields = cls._filter_outliers(city_emission_fields)
        return cls(city_emission_fields)

    @classmethod
    def _load_data(cls, tno_data: pl.DataFrame) -> list[CityEmissionField]:
        city_emission_fields = []
        cities = list(tno_data["City"].unique(maintain_order=True))
        for city in tqdm(cities, desc="Loading Cities", leave=False):
            city_data = tno_data.filter(pl.col("City") == city)
            original = CityEmissionField(city_data=city_data)
            city_emission_fields.append(original)
        return city_emission_fields

    @classmethod
    def _filter_outliers(cls, city_emission_fields: list[CityEmissionField]) -> list[CityEmissionField]:
        lower_bound = 0.001
        upper_bound = 0.025

        original_cities = {city.city_name for city in city_emission_fields}
        city_emission_fields = [c for c in city_emission_fields if np.percentile(c.co2_ff_field, 75) < upper_bound]
        cities_after_upper_bound_removed = {city.city_name for city in city_emission_fields}
        difference = original_cities - cities_after_upper_bound_removed
        logger.warning(f"Removed {difference} due to exceeding the upper bound (in total: {len(difference)})!")

        city_emission_fields = [c for c in city_emission_fields if np.percentile(c.co2_ff_field, 75) > lower_bound]
        cities_after_lower_bound_removed = {city.city_name for city in city_emission_fields}
        difference = cities_after_upper_bound_removed - cities_after_lower_bound_removed
        logger.warning(f"Removed {difference} due to exceeding the lower bound (in total: {len(difference)})!")

        return city_emission_fields

    def disable_temporal_transforms(self) -> None:
        self._temporal_transforms_enabled = False

    def add_sampling_transform(self, transform: EmissionFieldTransform) -> None:
        self._sampling_transforms.append(transform)

    def _initialize_transforms(self) -> None:
        self._hour_transforms = [HourTransform(hour=i) for i in range(1, 25)]
        self._day_transforms = [DayTransform(week_day=day) for day in Weekday]
        self._month_transforms = [MonthTransform(month=month) for month in Month]

    def _compute_number_of_emission_field_variations(self) -> int:
        return (
            len(self.city_emission_fields)
            * len(self._hour_transforms)
            * len(self._day_transforms)
            * len(self._month_transforms)
        )

    def __len__(self) -> int:
        if self._temporal_transforms_enabled:
            return self._compute_number_of_emission_field_variations()
        return len(self.city_emission_fields)

    def _validate_index_is_in_range(self, index: int) -> None:
        if not 0 <= index < len(self):
            key_error = f"Index {index} out of range!"
            raise IndexError(key_error)

    def _compose_temporal_transform_based_on_index(self, index: int) -> Compose:
        hour_transform_index = index % len(self._hour_transforms)
        remaining = index // len(self._hour_transforms)
        day_transform_index = remaining % len(self._day_transforms)
        remaining = remaining // len(self._day_transforms)
        month_transform_index = remaining % len(self._month_transforms)
        return Compose(
            [
                self._hour_transforms[hour_transform_index],
                self._day_transforms[day_transform_index],
                self._month_transforms[month_transform_index],
            ],
        )

    def _apply_temporal_transform_based_on_index(self, field: CityEmissionField, index: int) -> CityEmissionField:
        remaining = index // len(self.city_emission_fields)
        transform = self._compose_temporal_transform_based_on_index(index=remaining)
        return transform(field)

    def _get_emission_field_copy_at_index(self, index: int) -> CityEmissionField:
        self._validate_index_is_in_range(index)

        field = self.city_emission_fields[index % len(self.city_emission_fields)]

        copy = deepcopy(field)

        if self._temporal_transforms_enabled:
            self._apply_temporal_transform_based_on_index(field=copy, index=index)

        return copy

    def _apply_sampling_transform(self, city_emission_field: CityEmissionField) -> CityEmissionField:
        transform = Compose(self._sampling_transforms)
        return transform(city_emission_field)

    def get_city_emission_field(self, index: int, apply_sampling_transforms: bool = False) -> CityEmissionField:
        emission_field = self._get_emission_field_copy_at_index(index)
        if apply_sampling_transforms:
            self._apply_sampling_transform(emission_field)
        return emission_field

    def __getitem__(self, index: int) -> Tensor:
        emission_field = self._get_emission_field_copy_at_index(index)
        emission_field = self._apply_sampling_transform(emission_field)
        return emission_field.co2_ff_tensor
