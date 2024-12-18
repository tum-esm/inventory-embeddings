import numpy as np
import polars as pl
import torch
from torch import Tensor

from src.common.gnfr_sector import NUM_GNFR_SECTORS


class CityEmissionField:
    UNIT_CONVERSION_FACTOR = 1 / 1_387_584  # kg / km^2 * a -> μmol / m^2 * s (CO2)
    ROBUST_SCALING_FACTOR = 1 / 2_500_000  # determined using average of 95th percentile per city in training data

    def __init__(self, city_data: pl.DataFrame, year: int) -> None:
        """
        An emission field have dimension SxHxW (typical representations of images in numpy and torch).

        S: Number of sectors (for GNFR 15)
        H: Height
        W: Width

        The corresponding latitude and longitude array has dimensions WxH.
        """
        self.year = year

        self._name = city_data["City"][0]
        width: int = city_data["x"].max() + 1  # type: ignore[operator, assignment]
        height: int = city_data["y"].max() + 1  # type: ignore[operator, assignment]

        self.co2_ff_area_sources_field = np.zeros((NUM_GNFR_SECTORS, height, width))
        self.co2_ff_point_sources_field = np.zeros((NUM_GNFR_SECTORS, height, width))
        self.lat_lon_array = np.zeros((width, height, 2))

        for p in city_data.iter_rows(named=True):
            self.lat_lon_array[(p["x"], p["y"])] = [p["lat"], p["lon"]]
            for i, co_2_ff_sector in enumerate(p["co2_ff_area"].split(",")):
                self.co2_ff_area_sources_field[i, p["y"], p["x"]] = float(co_2_ff_sector)
            for i, co_2_ff_sector in enumerate(p["co2_ff_point"].split(",")):
                self.co2_ff_point_sources_field[i, p["y"], p["x"]] = float(co_2_ff_sector)
        self.co2_ff_area_sources_field *= self.ROBUST_SCALING_FACTOR
        self.co2_ff_point_sources_field *= self.ROBUST_SCALING_FACTOR

    @property
    def width(self) -> int:
        return self.co2_ff_area_sources_field.shape[2]

    @property
    def height(self) -> int:
        return self.co2_ff_area_sources_field.shape[1]

    @property
    def city_name(self) -> str:
        return self._name

    @property
    def co2_ff_area_sources_tensor(self) -> Tensor:
        return torch.tensor(self.co2_ff_area_sources_field.copy(), dtype=torch.float32)

    @property
    def co2_ff_point_sources_tensor(self) -> Tensor:
        return torch.tensor(self.co2_ff_point_sources_field.copy(), dtype=torch.float32)
