from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize
from matplotlib.image import AxesImage
from torch import Tensor

from src.common.constants import LON_LAT_ASPECT_RATIO
from src.common.gnfr_sector import GnfrSector
from src.dataset.city_emission_field import CityEmissionField

_TOTAL_EMISSIONS_ERROR = "Cannot provide a sector when plotting total emissions."
_TWO = 2
_CMAP = "viridis"
_SCALING_FACTOR = 1 / CityEmissionField.ROBUST_SCALING_FACTOR


def plot_emission_field(
    emission_field: CityEmissionField,
    ax: Axes,
    vmax: float | None = None,
    sector: GnfrSector | None = None,
    log_norm: bool = False,
    color_bar: bool = True,
    scale_to_real_emissions: bool = True,
) -> AxesImage:
    field = emission_field.co2_ff_area_sources_field
    to_plot = field[sector.to_index(), :, :] if sector else field.sum(0)

    scale = _SCALING_FACTOR if scale_to_real_emissions else 1

    bl_corner = emission_field.lat_lon_array[0, emission_field.height - 1]
    tr_corner = emission_field.lat_lon_array[emission_field.width - 1, 0]

    cmap = plt.get_cmap(_CMAP)
    cmap.set_bad(color=cmap(0))

    if log_norm:
        norm = LogNorm(vmin=scale * 1e-3 if vmax else None, vmax=scale * vmax if vmax else None)
    else:
        norm = Normalize(vmin=0 if vmax else None, vmax=scale * vmax if vmax else None)

    im = ax.imshow(
        scale * to_plot,
        cmap=cmap,
        extent=(float(bl_corner[1]), float(tr_corner[1]), float(bl_corner[0]), float(tr_corner[0])),
        aspect=LON_LAT_ASPECT_RATIO,
        norm=norm,
    )
    if color_bar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    city_name = emission_field.city_name
    title = f"{city_name}; {sector}" if sector else f"{city_name}; sum of all sectors"
    ax.set_title(title)

    return im


def plot_emission_field_tensor(
    emission_field: Tensor,
    ax: Axes,
    vmax: float | None = None,
    sector: GnfrSector | None = None,
    log_norm: bool = False,
    color_bar: bool = True,
    scale_to_real_emissions: bool = True,
) -> AxesImage:
    if emission_field.ndimension() == _TWO:
        if sector is not None:
            raise AssertionError(_TOTAL_EMISSIONS_ERROR)
        to_plot = emission_field
    else:
        to_plot = emission_field[sector.to_index(), :, :] if sector else emission_field.sum(0)

    scale = _SCALING_FACTOR if scale_to_real_emissions else 1

    cmap = plt.get_cmap(_CMAP)
    cmap.set_bad(color=cmap(0))

    if log_norm:
        norm = LogNorm(vmin=scale * 1e-3 if vmax else None, vmax=scale * vmax if vmax else None)
    else:
        norm = Normalize(vmin=0 if vmax else None, vmax=scale * vmax if vmax else None)

    im = ax.imshow(
        scale * to_plot,
        cmap=cmap,
        norm=norm,
    )

    if color_bar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return im
