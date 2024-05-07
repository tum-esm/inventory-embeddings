from matplotlib.axes import Axes
from matplotlib.pyplot import colormaps
from torch import Tensor

from embeddings.common.constants import LON_LAT_ASPECT_RATIO
from embeddings.common.gnfr_sector import GnfrSector
from embeddings.dataset.city_emission_field import CityEmissionField


def plot_emission_field_sector(
    emission_field: CityEmissionField,
    ax: Axes,
    sector: GnfrSector | None = None,  # None corresponds to summing the sectors
) -> None:
    field = emission_field.co2_ff_field_per_sector
    to_plot = field[sector.to_index(), :, :] if sector else field.sum(0)

    bl_corner = emission_field.lat_lon_array[0, emission_field.height - 1]
    tr_corner = emission_field.lat_lon_array[emission_field.width - 1, 0]

    ax.imshow(
        to_plot,
        cmap=colormaps["viridis"],
        extent=(float(bl_corner[1]), float(tr_corner[1]), float(bl_corner[0]), float(tr_corner[0])),
        aspect=LON_LAT_ASPECT_RATIO,
    )

    city_name = emission_field.city_name
    title = f"{city_name}; {sector}" if sector else f"{city_name}; sum of all sectors"
    ax.set_title(title)


def plot_emission_field_tensor(
    emission_field: Tensor,
    ax: Axes,
    vmax: float | None = None,
) -> None:
    ax.imshow(
        emission_field,
        cmap=colormaps["viridis"],
        vmin=None if vmax is None else 0,
        vmax=vmax,
    )
