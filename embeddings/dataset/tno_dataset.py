from copy import deepcopy
from pathlib import Path
from typing import Self

import polars as pl
from alive_progress import alive_bar
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from embeddings.common.log import logger
from embeddings.dataset.city_emission_field import CityEmissionField
from embeddings.dataset.emission_field_transforms import (
    CropTransform,
    DayTransform,
    HourTransform,
    Month,
    MonthTransform,
    Weekday,
)


class TnoDataset(Dataset[CityEmissionField]):
    def __init__(self, city_emission_fields: list[CityEmissionField]) -> None:
        self.city_emission_fields = city_emission_fields

        self._initialize_transforms()

    @classmethod
    def from_csv(cls, path: Path) -> Self:
        logger.info(f"Loading TNO data from '{path}'")
        tno_data = pl.read_csv(path, separator=";")
        city_emission_fields = cls._load_data(tno_data)
        return TnoDataset(city_emission_fields)

    @classmethod
    def _load_data(cls, tno_data: pl.DataFrame) -> list[CityEmissionField]:
        city_emission_fields = []
        cities = list(tno_data["City"].unique(maintain_order=True))
        with alive_bar(len(cities)) as bar:
            for city in cities:
                city_data = tno_data.filter(pl.col("City") == city)
                original = CityEmissionField(city_data=city_data)
                city_emission_fields.append(original)
                bar()
        return city_emission_fields

    def _initialize_transforms(self) -> None:
        self._crop_transforms = [
            CropTransform(center_offset_x=0, center_offset_y=0, width=32, height=32),
            CropTransform(center_offset_x=10, center_offset_y=10, width=32, height=32),
            CropTransform(center_offset_x=-10, center_offset_y=10, width=32, height=32),
            CropTransform(center_offset_x=10, center_offset_y=-10, width=32, height=32),
            CropTransform(center_offset_x=-10, center_offset_y=-10, width=32, height=32),
        ]
        self._hour_transforms = [HourTransform(hour=i) for i in range(1, 25)]
        self._day_transforms = [DayTransform(week_day=day) for day in Weekday]
        self._month_transforms = [MonthTransform(month=month) for month in Month]

    def _compute_number_of_emission_field_variations(self) -> int:
        return (
            len(self.city_emission_fields)
            * len(self._crop_transforms)
            * len(self._hour_transforms)
            * len(self._day_transforms)
            * len(self._month_transforms)
        )

    def __len__(self) -> int:
        return self._compute_number_of_emission_field_variations()

    def _validate_index_is_in_range(self, index: int) -> None:
        if not 0 <= index < len(self):
            key_error = f"Index {index} out of range!"
            raise IndexError(key_error)

    def _compose_transform_based_on_index(self, index: int) -> Compose:
        crop_transform_index = index % len(self._crop_transforms)
        remaining = index // len(self._crop_transforms)
        hour_transform_index = remaining % len(self._hour_transforms)
        remaining = remaining // len(self._hour_transforms)
        day_transform_index = remaining % len(self._day_transforms)
        remaining = remaining // len(self._day_transforms)
        month_transform_index = remaining % len(self._month_transforms)
        return Compose(
            [
                self._crop_transforms[crop_transform_index],
                self._hour_transforms[hour_transform_index],
                self._day_transforms[day_transform_index],
                self._month_transforms[month_transform_index],
            ],
        )

    def __getitem__(self, index: int) -> CityEmissionField:
        self._validate_index_is_in_range(index)

        original_data = self.city_emission_fields[index % len(self.city_emission_fields)]

        remaining = index // len(self.city_emission_fields)
        transform = self._compose_transform_based_on_index(index=remaining)

        copy = deepcopy(original_data)
        transform(copy)
        return copy
