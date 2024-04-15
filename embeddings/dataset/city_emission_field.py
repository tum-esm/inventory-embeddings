import numpy as np
import polars as pl
import torch
from matplotlib.pyplot import Axes, colormaps
from torch import Tensor

from embeddings.common.constants import LON_LAT_ASPECT_RATIO
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
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def city_name(self) -> str:
        return self._name

    @property
    def co2_ff_tensor(self) -> Tensor:
        return self._co2_ff_tensor

    @co2_ff_tensor.setter
    def co2_ff_tensor(self, value: Tensor) -> None:
        self._width = value.shape[0]
        self._height = value.shape[1]
        self._co2_ff_tensor = value

    @property
    def lat_lon_array(self) -> np.array:
        return self._lat_lon_array

    @lat_lon_array.setter
    def lat_lon_array(self, value: Tensor) -> None:
        self._lat_lon_array = value

    def plot(self, ax: Axes, sector: GnfrSector | None = None) -> None:
        to_plot = self._co2_ff_tensor[:, :, sector.to_index()] if sector else self._co2_ff_tensor.sum(2)

        bl_corner = self._lat_lon_array[0, self._height - 1]
        tr_corner = self._lat_lon_array[self._width - 1, 0]

        ax.imshow(
            to_plot.T,
            cmap=colormaps["viridis"],
            extent=(float(bl_corner[1]), float(tr_corner[1]), float(bl_corner[0]), float(tr_corner[0])),
            aspect=LON_LAT_ASPECT_RATIO,
        )

        title = f"{self._name}; {sector}" if sector else f"{self._name}; sum of all sectors"
        ax.set_title(title)
