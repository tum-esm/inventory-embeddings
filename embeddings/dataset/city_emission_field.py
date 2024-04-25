import numpy as np
import polars as pl
import torch
from torch import Tensor

from embeddings.common.gnfr_sector import NUM_GNFR_SECTORS


class CityEmissionField:
    def __init__(self, city_data: pl.DataFrame) -> None:
        """
        An emission field have dimension SxWxH.

        S: Number of sectors (for GNFR 15)
        W: Width
        H: Height
        """

        self._name = city_data["City"][0]
        width: int = city_data["x"].max() + 1  # type: ignore[operator, assignment]
        height: int = city_data["y"].max() + 1  # type: ignore[operator, assignment]

        self.co2_ff_field = np.zeros((NUM_GNFR_SECTORS, width, height))
        self.lat_lon_array = np.zeros((width, height, 2))

        for p in city_data.iter_rows(named=True):
            self.lat_lon_array[(p["x"], p["y"])] = [p["lat"], p["lon"]]
            for i, co_2ff_sector in enumerate(p["co2_ff"].split(",")):
                self.co2_ff_field[i, p["x"], p["y"]] = float(co_2ff_sector) / 1000

    @property
    def width(self) -> int:
        return self.co2_ff_field.shape[1]

    @property
    def height(self) -> int:
        return self.co2_ff_field.shape[2]

    @property
    def city_name(self) -> str:
        return self._name

    @property
    def co2_ff_tensor(self) -> Tensor:
        return torch.tensor(self.co2_ff_field.copy(), dtype=torch.float32)
