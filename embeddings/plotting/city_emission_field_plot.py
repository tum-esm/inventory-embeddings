from matplotlib.axes import Axes
from matplotlib.pyplot import colormaps
from torch import Tensor

from embeddings.common.constants import LON_LAT_ASPECT_RATIO
from embeddings.common.gnfr_sector import GnfrSector
from embeddings.dataset.city_emission_field import CityEmissionField

_TOTAL_EMISSIONS_ERROR = "Cannot provide a sector when plotting total emissions."
_TWO = 2


def plot_emission_field(
    emission_field: CityEmissionField,
    ax: Axes,
    vmax: float | None = None,
    sector: GnfrSector | None = None,
) -> None:
    field = emission_field.co2_ff_field
    to_plot = field[sector.to_index(), :, :] if sector else field.sum(0)

    bl_corner = emission_field.lat_lon_array[0, emission_field.height - 1]
    tr_corner = emission_field.lat_lon_array[emission_field.width - 1, 0]

    ax.imshow(
        to_plot,
        cmap=colormaps["viridis"],
        extent=(float(bl_corner[1]), float(tr_corner[1]), float(bl_corner[0]), float(tr_corner[0])),
        aspect=LON_LAT_ASPECT_RATIO,
        vmin=None if vmax is None else 0,
        vmax=vmax,
    )

    city_name = emission_field.city_name
    title = f"{city_name}; {sector}" if sector else f"{city_name}; sum of all sectors"
    ax.set_title(title)


def plot_emission_field_tensor(
    emission_field: Tensor,
    ax: Axes,
    vmax: float | None = None,
    sector: GnfrSector | None = None,
) -> None:
    if emission_field.ndimension() == _TWO:
        if sector is not None:
            raise AssertionError(_TOTAL_EMISSIONS_ERROR)
        to_plot = emission_field
    else:
        to_plot = emission_field[sector.to_index(), :, :] if sector else emission_field.sum(0)

    ax.imshow(
        to_plot,
        cmap=colormaps["viridis"],
        vmin=None if vmax is None else 0,
        vmax=vmax,
    )
