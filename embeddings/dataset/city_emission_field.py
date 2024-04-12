import polars as pl
import torch
from matplotlib.pyplot import Axes, colormaps
from torch import Tensor

from embeddings.common.gnfr_sector import NUM_GNFR_SECTORS, GnfrSector


class CityEmissionField:
    def __init__(self, city_data: pl.DataFrame) -> None:
        self._name = city_data["City"][0]
        self._max_grid_width = city_data["x"].max() + 1
        self._max_grid_height = city_data["y"].max() + 1

        self._pixel_coordinates_to_lat_lon_map = {}

        self._co2_ff_tensor = torch.zeros(self._max_grid_width, self._max_grid_height, NUM_GNFR_SECTORS)

        for p in city_data.iter_rows(named=True):
            self._pixel_coordinates_to_lat_lon_map[(p["x"], p["y"])] = (p["lat"], p["lon"])
            for i, co_2ff_sector in enumerate(p["co2_ff"].split(",")):
                self._co2_ff_tensor[p["x"], p["y"], i] = float(co_2ff_sector)

    @property
    def city_name(self) -> str:
        return self._name

    @property
    def as_tensor(self) -> Tensor:
        return self._co2_ff_tensor

    def plot(self, ax: Axes, sector: GnfrSector | None = None) -> None:
        to_plot = self._co2_ff_tensor[:, :, sector.to_index()] if sector else self._co2_ff_tensor.sum(2)

        tl_corner = self._pixel_coordinates_to_lat_lon_map[(0, 0)]
        br_corner = self._pixel_coordinates_to_lat_lon_map[(self._max_grid_width - 1, self._max_grid_height - 1)]

        aspect_ratio = 2 * self._max_grid_width / self._max_grid_width

        ax.imshow(
            to_plot.T,
            cmap=colormaps["plasma"],
            extent=(tl_corner[1], br_corner[1], br_corner[0], tl_corner[0]),
            aspect=aspect_ratio,
        )

        title = f"{self._name}; {sector}" if sector else f"{self._name}; sum of all sectors"
        ax.set_title(title)
