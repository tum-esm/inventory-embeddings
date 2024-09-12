import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from tqdm import tqdm

from src.common.constants import TNO_LAT_STEP, TNO_LON_STEP
from src.common.gnfr_sector import GnfrSector
from src.common.log import logger
from src.data_preprocessing.city_filtering import filter_cities_from_open_data_soft_data
from src.data_preprocessing.data_classes.cell import Cell, CellBuilder
from src.data_preprocessing.data_classes.city import City
from src.data_preprocessing.data_classes.ghg_source import GhgPointSource, GhgSource


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
        tno_data = pl.read_csv(tno_csv, separator=";", ignore_errors=True, truncate_ragged_lines=True)
        cities = self._filter_cities(tno_data=tno_data)

        self._write_csv_header(out_csv=out_csv)

        for city in tqdm(cities, desc="Processing", file=sys.stdout):
            cells = self._process_city(city, tno_data)
            self._write_cells_to_csv(out_csv=out_csv, city=city, cells=cells)

    def _process_city(self, city: City, tno_data: pl.DataFrame) -> list[Cell]:
        city_sources_data = self._filter_city_by_latitude_and_longitude(tno_data, city)
        cells = self._extract_cells_from_city_data(city_data=city_sources_data)

        self._log_warning_if_required(city=city, cells=cells)

        cells.sort(key=lambda c: (-c.lat, c.lon))

        return cells

    def _get_min_max(self, data: pl.DataFrame, column: str, margin: float) -> tuple[float, float]:
        return (
            data[column].min() + margin / 2,  # type: ignore[operator]
            data[column].max() - margin / 2,  # type: ignore[operator]
        )

    def _filter_cities(self, tno_data: pl.DataFrame) -> list[City]:
        return filter_cities_from_open_data_soft_data(
            min_population_size=self._options.min_population_size,
            lat_range=self._get_min_max(tno_data, column="Lat", margin=TNO_LAT_STEP * self._options.grid_height),
            lon_range=self._get_min_max(tno_data, column="Lon", margin=TNO_LON_STEP * self._options.grid_width),
            show_warnings=self._options.show_warnings,
        )

    def _write_csv_header(self, out_csv: Path) -> None:
        with out_csv.open("w") as file:
            csv_writer = csv.writer(file, delimiter=";")
            csv_writer.writerow(
                [
                    "City",
                    "x",
                    "y",
                    "lat",
                    "lon",
                    "co2_ff_area",
                    "co2_ff_point",
                    "co2_bf_area",
                    "co2_bf_point",
                    "ch4_area",
                    "ch4_point",
                ],
            )

    def _write_cells_to_csv(self, out_csv: Path, city: City, cells: list[Cell]) -> None:
        with out_csv.open("a") as file:
            csv_writer = csv.writer(file, delimiter=";")
            for cell in cells:
                csv_writer.writerow(
                    [
                        city.name,
                        cell.x,
                        cell.y,
                        cell.lat,
                        cell.lon,
                        cell.co2_ff_area_str,
                        cell.co2_ff_point_str,
                        cell.co2_bf_area_str,
                        cell.co2_bf_point_str,
                        cell.ch4_area_str,
                        cell.ch4_point_str,
                    ],
                )

    def _filter_city_by_latitude_and_longitude(self, tno_data: pl.DataFrame, city: City) -> pl.DataFrame:
        return tno_data.filter(
            (pl.col("Lat") >= city.lat - (self._options.grid_height / 2) * TNO_LAT_STEP)
            & (pl.col("Lat") <= city.lat + (self._options.grid_height / 2) * TNO_LAT_STEP)
            & (pl.col("Lon") >= city.lon - (self._options.grid_width / 2) * TNO_LON_STEP)
            & (pl.col("Lon") <= city.lon + (self._options.grid_width / 2) * TNO_LON_STEP),
        )

    def _log_warning_if_required(self, city: City, cells: list[Cell]) -> None:
        if self._options.show_warnings and len(cells) != self._options.grid_width * self._options.grid_height:
            logger.warning(
                f"Number of cells for {city.name} {city.lat, city.lon} does not match expected! "
                f"Expected:{self._options.grid_width * self._options.grid_height}; Got:{len(cells)}.\n",
            )

    def _extract_area_sources_from(self, sources_data: pl.DataFrame) -> list[GhgSource]:
        return [
            GhgSource(
                sector=GnfrSector.from_str(source["GNFR_Sector"]),
                co2_ff=source["CO2_ff"] if source["CO2_ff"] else 0.0,
                co2_bf=source["CO2_bf"] if source["CO2_bf"] else 0.0,
                ch4=source["CH4"] if source["CH4"] else 0.0,
            )
            for source in sources_data.iter_rows(named=True)
        ]

    def _extract_point_sources(self, sources_data: pl.DataFrame) -> list[GhgPointSource]:
        return [
            GhgPointSource(
                sector=GnfrSector.from_str(source["GNFR_Sector"]),
                co2_ff=source["CO2_ff"] if source["CO2_ff"] else 0.0,
                co2_bf=source["CO2_bf"] if source["CO2_bf"] else 0.0,
                ch4=source["CH4"] if source["CH4"] else 0.0,
                lat=source["Lat"],
                lon=source["Lon"],
            )
            for source in sources_data.iter_rows(named=True)
        ]

    def _assign_point_sources_to_nearest_cell(
        self,
        point_sources: list[GhgPointSource],
        cell_coordinates: list[tuple[float, float]],
    ) -> dict[tuple[float, float], list[GhgPointSource]]:
        lon_lat_point_source: dict[tuple[float, float], list[GhgPointSource]] = {}
        for point_source in point_sources:
            min_distance = float("inf")
            min_lon_lat = (-1.0, -1.0)
            for lon, lat in cell_coordinates:
                distance = (point_source.lat - lat) ** 2 + (point_source.lon - lon) ** 2
                if distance < min_distance:
                    min_distance = distance
                    min_lon_lat = (lon, lat)
            before = lon_lat_point_source.get(min_lon_lat, [])
            before.append(point_source)
            lon_lat_point_source[min_lon_lat] = before
        return lon_lat_point_source

    def _extract_cells_from_city_data(self, city_data: pl.DataFrame) -> list[Cell]:
        area_sources_data = city_data.filter(pl.col("SourceType") == "A")
        point_sources_data = city_data.filter(pl.col("SourceType") == "P")

        point_sources = self._extract_point_sources(sources_data=point_sources_data)

        min_lon = area_sources_data["Lon"].min()
        max_lat = area_sources_data["Lat"].max()

        cells_data = area_sources_data.group_by(["Lon", "Lat"])

        lon_lat = [lon_lat for lon_lat, _ in cells_data]
        lon_lat_point_source = self._assign_point_sources_to_nearest_cell(
            point_sources=point_sources,
            cell_coordinates=lon_lat,
        )

        cells = []

        for (lon, lat), data in cells_data:  # type: ignore[misc]
            area_sources = self._extract_area_sources_from(sources_data=data)
            point_sources = lon_lat_point_source.get((lon, lat), [])
            cell = (
                CellBuilder()
                .with_ghg_area_sources(area_sources)
                .with_ghg_point_sources(point_sources)
                .with_coordinates(lat=lat, lon=lon)  # type: ignore[arg-type]
                .with_coordinates_origin(lon=min_lon, lat=max_lat)  # type: ignore[arg-type]
                .build()
            )
            cells.append(cell)

        return cells
