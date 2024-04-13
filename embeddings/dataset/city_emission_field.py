import numpy as np
import polars as pl
import torch
from matplotlib.pyplot import Axes, colormaps
from torch import Tensor

from embeddings.common.gnfr_sector import NUM_GNFR_SECTORS, GnfrSector


class CityEmissionField:
    def __init__(self, city_data: pl.DataFrame) -> None:
        self._name = city_data["City"][0]
        self._width = city_data["x"].max() + 1
        self._height = city_data["y"].max() + 1

        self._co2_ff_tensor = torch.zeros(self._width, self._height, NUM_GNFR_SECTORS)
        self._lat_lon_array = np.zeros((self._width, self._height, 2))

        for p in city_data.iter_rows(named=True):
            self._lat_lon_array[(p["x"], p["y"])] = [p["lat"], p["lon"]]
            for i, co_2ff_sector in enumerate(p["co2_ff"].split(",")):
                self._co2_ff_tensor[p["x"], p["y"], i] = float(co_2ff_sector)

    @property
    def city_name(self) -> str:
        return self._name

    @property
    def as_tensor(self) -> Tensor:
        return self._co2_ff_tensor

    def crop(
        self,
        center_offset_x: int,
        center_offset_y: int,
        width: int,
        height: int,
    ) -> None:
        center_x = self._width // 2 + center_offset_x
        center_y = self._height // 2 + center_offset_y

        self._width = width
        self._height = height

        start_x = center_x - width // 2
        start_y = center_y - height // 2
        end_x = start_x + width
        end_y = start_y + height

        self._co2_ff_tensor = self._co2_ff_tensor[start_x:end_x, start_y:end_y, :]
        self._lat_lon_array = self._lat_lon_array[start_x:end_x, start_y:end_y, :]

    def plot(self, ax: Axes, sector: GnfrSector | None = None) -> None:
        to_plot = self._co2_ff_tensor[:, :, sector.to_index()] if sector else self._co2_ff_tensor.sum(2)

        tl_corner = self._lat_lon_array[0, 0]
        br_corner = self._lat_lon_array[self._width - 1, self._height - 1]

        ax.imshow(
            to_plot.T,
            cmap=colormaps["plasma"],
            extent=(float(tl_corner[1]), float(br_corner[1]), float(br_corner[0]), float(tl_corner[0])),
            aspect=2,
        )

        title = f"{self._name}; {sector}" if sector else f"{self._name}; sum of all sectors"
        ax.set_title(title)
