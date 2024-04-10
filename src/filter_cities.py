import csv
from dataclasses import dataclass
from math import floor
from pathlib import Path

import polars as pl
from alive_progress import alive_bar

from src.constants import TNO_LAT_STEP, TNO_LONG_STEP
from src.gnfr_sector_type import GnfrSectorType
from src.helper_dataclasses import Cell, GhgSource


@dataclass
class _City:
    name: str
    lat: float
    lon: float


def _get_all_cities(
    min_population_size: int,  # noqa: ARG001
    min_lat: float,  # noqa: ARG001
    max_lat: float,  # noqa: ARG001
    min_lon: float,  # noqa: ARG001
    max_lon: float,  # noqa: ARG001
) -> list[_City]:
    return [
        _City("Munich", 48.13743, 11.57549),
        _City("Paris", 48.85341, 2.3488),
    ]


def filter_cities(
    tno_data_csv: Path,
    out_csv: Path,
    *,
    grid_width: int = 61,
    grid_height: int = 61,
    min_population_size: int = 1_000_000,
) -> None:
    tno_data = pl.read_csv(tno_data_csv, separator=";")

    cites = _get_all_cities(
        min_population_size=min_population_size,
        min_lat=0,
        max_lat=99,
        min_lon=0,
        max_lon=99,
    )
    with out_csv.open("w") as file:
        csv_writer = csv.writer(file, delimiter=";")
        csv_writer.writerow(["City", "x", "y", "co2_ff", "co2_bf", "ch4"])

    with alive_bar(total=len(cites)) as bar:
        for city in cites:
            city_sources_data = tno_data.filter(
                (pl.col("Lat") >= city.lat - (grid_height / 2) * TNO_LAT_STEP)
                & (pl.col("Lat") <= city.lat + (grid_height / 2) * TNO_LAT_STEP)
                & (pl.col("Lon") >= city.lon - (grid_width / 2) * TNO_LONG_STEP)
                & (pl.col("Lon") <= city.lon + (grid_width / 2) * TNO_LONG_STEP),
            )
            # Important: must also check for point sources
            city_area_sources_data = city_sources_data.filter(pl.col("SourceType") == "A")
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

            if len(cells) != grid_width * grid_height:
                exception_text = "This data does not fit"
                raise ValueError(exception_text)

            cells.sort(key=lambda c: (-c.lat, c.lon))

            with out_csv.open("a") as file:
                csv_writer = csv.writer(file, delimiter=";")
                for i, cell in enumerate(cells):
                    y = floor(i / grid_width)
                    x = i - y * grid_width
                    csv_writer.writerow([city.name, x, y, cell.co2_ff_str, cell.co2_bf_str, cell.ch4_str])

            bar()
