import csv
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from alive_progress import alive_bar

from embeddings.common.constants import TNO_LAT_STEP, TNO_LON_STEP
from embeddings.common.gnfr_sector import GnfrSector
from embeddings.data_preprocessing.city_filtering import filter_cities_from_open_data_soft_data
from embeddings.data_preprocessing.data_classes.cell import Cell, CellBuilder
from embeddings.data_preprocessing.data_classes.city import City
from embeddings.data_preprocessing.data_classes.ghg_source import GhgSource


@dataclass
class TnoPreprocessorOptions:
    grid_width: int = 61
    grid_height: int = 61
    min_population_size: int = 500_000
    show_warnings: bool = False


class TnoPreprocessor:
    def __init__(self, options: TnoPreprocessorOptions) -> None:
        self._options = options

    def preprocess(self, tno_csv: Path, out_csv: Path) -> None:
        tno_data = pl.read_csv(tno_csv, separator=";")
        cites = self._filter_cities(tno_data=tno_data)

        self._write_csv_header(out_csv=out_csv)

        with alive_bar(total=len(cites)) as bar:
            for city in cites:
                cells = self._process_city(city, tno_data)
                self._write_cells_to_csv(out_csv=out_csv, city=city, cells=cells)
                bar()

    def _process_city(self, city: City, tno_data: pl.DataFrame) -> list[Cell]:
        city_sources_data = self._filter_city_by_latitude_and_longitude(tno_data, city)
        cells = self._extract_cells_from_city_data(city_data=city_sources_data)

        self._print_warning_if_required(city=city, cells=cells)

        cells.sort(key=lambda c: (-c.lat, c.lon))

        return cells

    def _get_min_max(self, data: pl.DataFrame, column: str, margin: float) -> tuple[float, float]:
        return (
            data[column].min() + margin / 2,
            data[column].max() - margin / 2,
        )

    def _filter_cities(self, tno_data: pl.DataFrame) -> list[City]:
        return filter_cities_from_open_data_soft_data(
            min_population_size=self._options.min_population_size,
            lat_range=self._get_min_max(tno_data, column="Lat", margin=TNO_LAT_STEP * self._options.grid_height),
            lon_range=self._get_min_max(tno_data, column="Lon", margin=TNO_LON_STEP * self._options.grid_width),
        )

    def _write_csv_header(self, out_csv: Path) -> None:
        with out_csv.open("w") as file:
            csv_writer = csv.writer(file, delimiter=";")
            csv_writer.writerow(["City", "x", "y", "lat", "lon", "co2_ff", "co2_bf", "ch4"])

    def _write_cells_to_csv(self, out_csv: Path, city: City, cells: list[Cell]) -> None:
        with out_csv.open("a") as file:
            csv_writer = csv.writer(file, delimiter=";")
            for cell in cells:
                csv_writer.writerow(
                    [city.name, cell.x, cell.y, cell.lat, cell.lon, cell.co2_ff_str, cell.co2_bf_str, cell.ch4_str],
                )

    def _filter_city_by_latitude_and_longitude(self, tno_data: pl.DataFrame, city: City) -> pl.DataFrame:
        return tno_data.filter(
            (pl.col("Lat") >= city.lat - (self._options.grid_height / 2) * TNO_LAT_STEP)
            & (pl.col("Lat") <= city.lat + (self._options.grid_height / 2) * TNO_LAT_STEP)
            & (pl.col("Lon") >= city.lon - (self._options.grid_width / 2) * TNO_LON_STEP)
            & (pl.col("Lon") <= city.lon + (self._options.grid_width / 2) * TNO_LON_STEP),
        )

    def _print_warning_if_required(self, city: City, cells: list[Cell]) -> None:
        if self._options.show_warnings and len(cells) != self._options.grid_width * self._options.grid_height:
            print(
                f"Warning: Number of cells for {city.name} {city.lat, city.lon} does not match expected! "
                f"Expected:{self._options.grid_width * self._options.grid_height}; Got:{len(cells)}.\n",
            )

    def _extract_sources_from(self, sources_data: pl.DataFrame) -> list[GhgSource]:
        return [
            GhgSource(
                sector=GnfrSector.from_str(source["GNFR_Sector"]),
                co2_ff=source["CO2_ff"],
                co2_bf=source["CO2_bf"],
                ch4=source["CH4"],
            )
            for source in sources_data.iter_rows(named=True)
        ]

    def _extract_cells_from_city_data(self, city_data: pl.DataFrame) -> list[Cell]:
        # TODO: must also check for point sources  # noqa: FIX002, TD002, TD003
        city_area_sources_data = city_data.filter(pl.col("SourceType") == "A")

        min_lon = city_area_sources_data["Lon"].min()
        max_lat = city_area_sources_data["Lat"].max()

        cells_data = city_area_sources_data.group_by(["Lon", "Lat"])

        cells = []

        for (lon, lat), data in cells_data:
            sources = self._extract_sources_from(sources_data=data)
            cell = (
                CellBuilder()
                .with_ghg_sources(sources)
                .with_coordinates(lat=lat, lon=lon)
                .with_coordinates_origin(lon=min_lon, lat=max_lat)
                .build()
            )
            cells.append(cell)
        return cells
