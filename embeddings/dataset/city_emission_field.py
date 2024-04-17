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
        width = city_data["x"].max() + 1
        height = city_data["y"].max() + 1

        self.co2_ff_field = np.zeros((width, height, NUM_GNFR_SECTORS))
        self.lat_lon_array = np.zeros((width, height, 2))

        for p in city_data.iter_rows(named=True):
            self.lat_lon_array[(p["x"], p["y"])] = [p["lat"], p["lon"]]
            for i, co_2ff_sector in enumerate(p["co2_ff"].split(",")):
                self.co2_ff_field[p["x"], p["y"], i] = float(co_2ff_sector)

    @property
    def width(self) -> int:
        return self.co2_ff_field.shape[0]

    @property
    def height(self) -> int:
        return self.co2_ff_field.shape[1]

    @property
    def city_name(self) -> str:
        return self._name

    @property
    def co2_ff_tensor(self) -> Tensor:
        return torch.tensor(self.co2_ff_field)

    def plot(self, ax: Axes, sector: GnfrSector | None = None) -> None:
        to_plot = self.co2_ff_field[:, :, sector.to_index()] if sector else self.co2_ff_field.sum(2)

        bl_corner = self.lat_lon_array[0, self.height - 1]
        tr_corner = self.lat_lon_array[self.width - 1, 0]

        ax.imshow(
            to_plot.T,
            cmap=colormaps["viridis"],
            extent=(float(bl_corner[1]), float(tr_corner[1]), float(bl_corner[0]), float(tr_corner[0])),
            aspect=LON_LAT_ASPECT_RATIO,
        )

        title = f"{self._name}; {sector}" if sector else f"{self._name}; sum of all sectors"
        ax.set_title(title)
