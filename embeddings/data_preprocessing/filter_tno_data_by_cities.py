import csv
from dataclasses import dataclass
from math import floor
from pathlib import Path

import polars as pl
from alive_progress import alive_bar

from embeddings.common.paths import OpenDataSoftPaths
from embeddings.data_preprocessing.constants import TNO_LAT_STEP, TNO_LON_STEP
from embeddings.data_preprocessing.gnfr_sector_type import GnfrSectorType
from embeddings.data_preprocessing.helper_dataclasses import Cell, GhgSource


@dataclass
class _City:
    name: str
    lat: float
    lon: float


def _coordinates_to_float(coordinates: str) -> tuple[float, float]:
    lat, lon = coordinates.split(",")
    return float(lat), float(lon)


def _are_coordinates_within_range(
    coordinates: str,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
) -> bool:
    lat, lon = _coordinates_to_float(coordinates)
    return lat_range[0] <= lat <= lat_range[1] and lon_range[0] <= lon <= lon_range[1]


def _get_all_cities(
    min_population_size: int,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
) -> list[_City]:
    res = pl.read_csv(OpenDataSoftPaths.OPEN_DATA_SOFT_GEONAMES_CSV, separator=";", infer_schema_length=10000)

    filtered_cities = res.filter(
        (pl.col("Population") >= min_population_size),
        (
            pl.col("Coordinates").map_elements(
                function=lambda val: _are_coordinates_within_range(val, lat_range=lat_range, lon_range=lon_range),
                return_dtype=bool,
            ),
        ),
    )

    cities = []

    for city in filtered_cities.iter_rows(named=True):
        lat, lon = _coordinates_to_float(city["Coordinates"])
        cities.append(_City(name=city["Name"], lat=lat, lon=lon))

    return cities


def _filter_city_by_latitude_and_longitude(
    tno_data: pl.DataFrame,
    city: _City,
    height: int,
    width: int,
) -> pl.DataFrame:
    return tno_data.filter(
        (pl.col("Lat") >= city.lat - (height / 2) * TNO_LAT_STEP)
        & (pl.col("Lat") <= city.lat + (height / 2) * TNO_LAT_STEP)
        & (pl.col("Lon") >= city.lon - (width / 2) * TNO_LON_STEP)
        & (pl.col("Lon") <= city.lon + (width / 2) * TNO_LON_STEP),
    )


def _extract_cells_from_city_data(
    city_data: pl.DataFrame,
) -> list[Cell]:
    # TODO: must also check for point sources  # noqa: FIX002, TD002, TD003
    city_area_sources_data = city_data.filter(pl.col("SourceType") == "A")
    cells_data = city_area_sources_data.group_by(["Lon", "Lat"])

    cells = []

    for (lon, lat), data in cells_data:
        sources = [
            GhgSource(
                sector=GnfrSectorType.from_str(source["GNFR_Sector"]),
                co2_ff=source["CO2_ff"],
                co2_bf=source["CO2_bf"],
                ch4=source["CH4"],
            )
            for source in data.iter_rows(named=True)
        ]

        cells.append(Cell.from_ghg_sources(lon, lat, sources))
    return cells


def _write_csv_header(out_csv: Path) -> None:
    with out_csv.open("w") as file:
        csv_writer = csv.writer(file, delimiter=";")
        csv_writer.writerow(["City", "x", "y", "co2_ff", "co2_bf", "ch4"])


def _write_cells_to_csv(out_csv: Path, city: _City, cells: dict[tuple[int, int], Cell]) -> None:
    with out_csv.open("a") as file:
        csv_writer = csv.writer(file, delimiter=";")
        for (x, y), cell in cells.items():
            csv_writer.writerow([city.name, x, y, cell.co2_ff_str, cell.co2_bf_str, cell.ch4_str])


def _convert_index_to_coordinates(index: int, grid_width: int) -> tuple[int, int]:
    y = floor(index / grid_width)
    x = index - y * grid_width
    return x, y


def filter_tno_data_by_cities(
    tno_data_csv: Path,
    out_csv: Path,
    *,
    grid_width: int = 61,
    grid_height: int = 61,
    min_population_size: int = 500_000,
) -> None:
    tno_data = pl.read_csv(tno_data_csv, separator=";")

    cites = _get_all_cities(
        min_population_size=min_population_size,
        lat_range=(
            tno_data["Lat"].min() + TNO_LAT_STEP * grid_height / 2,
            tno_data["Lat"].max() - TNO_LAT_STEP * grid_height / 2,
        ),
        lon_range=(
            tno_data["Lon"].min() + TNO_LON_STEP * grid_width / 2,
            tno_data["Lon"].max() - TNO_LON_STEP * grid_width / 2,
        ),
    )

    _write_csv_header(out_csv=out_csv)

    with alive_bar(total=len(cites)) as bar:
        for city in cites:
            city_sources_data = _filter_city_by_latitude_and_longitude(tno_data, city, grid_width, grid_height)
            cells = _extract_cells_from_city_data(city_data=city_sources_data)

            if len(cells) != grid_width * grid_height:
                warning = (
                    f"Warning: There are not enough cells in {city.name} {city.lat, city.lon}! "
                    f"Expected:{grid_width * grid_height}; Got:{len(cells)}.\n"
                    f"\tSkipping {city.name}"
                )
                print(warning)
                continue

            cells.sort(key=lambda c: (-c.lat, c.lon))

            coordinates_to_cell = {_convert_index_to_coordinates(i, grid_width): cell for i, cell in enumerate(cells)}

            _write_cells_to_csv(out_csv=out_csv, city=city, cells=coordinates_to_cell)

            bar()
