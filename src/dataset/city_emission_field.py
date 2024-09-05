import numpy as np
import polars as pl
import torch
from torch import Tensor

from src.common.gnfr_sector import NUM_GNFR_SECTORS


class CityEmissionField:
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

        self.co2_ff_field = np.zeros((NUM_GNFR_SECTORS, height, width))
        self.lat_lon_array = np.zeros((width, height, 2))

        for p in city_data.iter_rows(named=True):
            self.lat_lon_array[(p["x"], p["y"])] = [p["lat"], p["lon"]]
            for i, co_2ff_sector in enumerate(p["co2_ff"].split(",")):
                self.co2_ff_field[i, p["y"], p["x"]] = float(co_2ff_sector) * self.ROBUST_SCALING_FACTOR

    @property
    def width(self) -> int:
        return self.co2_ff_field.shape[2]

    @property
    def height(self) -> int:
        return self.co2_ff_field.shape[1]

    @property
    def city_name(self) -> str:
        return self._name

    @property
    def co2_ff_tensor(self) -> Tensor:
        return torch.tensor(self.co2_ff_field.copy(), dtype=torch.float32)
