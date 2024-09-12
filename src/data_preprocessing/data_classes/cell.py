from dataclasses import dataclass
from typing import Self

from src.common.constants import TNO_LAT_STEP, TNO_LON_STEP
from src.data_preprocessing.data_classes.ghg_source import GhgPointSource, GhgSource


@dataclass
class Cell:
    lat: float
    lon: float
    x: int
    y: int
    co2_ff_area: list[float]
    co2_ff_point: list[float]
    co2_bf_area: list[float]
    co2_bf_point: list[float]
    ch4_area: list[float]
    ch4_point: list[float]

    @property
    def co2_ff_area_str(self) -> str:
        return ",".join([str(val) for val in self.co2_ff_area])

    @property
    def co2_ff_point_str(self) -> str:
        return ",".join([str(val) for val in self.co2_ff_point])

    @property
    def co2_bf_area_str(self) -> str:
        return ",".join([str(val) for val in self.co2_bf_area])

    @property
    def co2_bf_point_str(self) -> str:
        return ",".join([str(val) for val in self.co2_bf_point])

    @property
    def ch4_area_str(self) -> str:
        return ",".join([str(val) for val in self.ch4_area])

    @property
    def ch4_point_str(self) -> str:
        return ",".join([str(val) for val in self.ch4_point])


class CellBuilder:
    def __init__(self) -> None:
        self._lat = 0.0
        self._lon = 0.0
        self._lon_or = 0.0
        self._lat_or = 0.0
        self._ghg_area_sources: list[GhgSource] = []
        self._ghg_point_sources: list[GhgPointSource] = []

    def with_coordinates(self, lat: float, lon: float) -> Self:
        self._lat = lat
        self._lon = lon
        return self

    def with_coordinates_origin(self, lon: float, lat: float) -> Self:
        self._lon_or = lon
        self._lat_or = lat
        return self

    def with_ghg_area_sources(self, sources: list[GhgSource]) -> Self:
        self._ghg_area_sources = sources
        return self

    def with_ghg_point_sources(self, sources: list[GhgPointSource]) -> Self:
        self._ghg_point_sources = sources
        return self

    def _compute_pixel_coordinates(self) -> tuple[int, int]:
        x = round((self._lon - self._lon_or) / TNO_LON_STEP)
        y = round((self._lat_or - self._lat) / TNO_LAT_STEP)
        return x, y

    def build(self) -> Cell:
        co2_ff_area_list = [0.0 for _ in range(15)]
        co2_bf_area_list = [0.0 for _ in range(15)]
        ch4_area_list = [0.0 for _ in range(15)]
        for source in self._ghg_area_sources:
            co2_ff_area_list[source.sector.to_index()] += source.co2_ff
            co2_bf_area_list[source.sector.to_index()] += source.co2_bf
            ch4_area_list[source.sector.to_index()] += source.ch4

        co2_ff_point_list = [0.0 for _ in range(15)]
        co2_bf_point_list = [0.0 for _ in range(15)]
        ch4_point_list = [0.0 for _ in range(15)]
        for source in self._ghg_point_sources:
            co2_ff_point_list[source.sector.to_index()] += source.co2_ff
            co2_bf_point_list[source.sector.to_index()] += source.co2_bf
            ch4_point_list[source.sector.to_index()] += source.ch4

        x, y = self._compute_pixel_coordinates()

        return Cell(
            lat=self._lat,
            lon=self._lon,
            x=x,
            y=y,
            co2_ff_area=co2_ff_area_list,
            co2_ff_point=co2_ff_point_list,
            co2_bf_area=co2_bf_area_list,
            co2_bf_point=co2_bf_point_list,
            ch4_area=ch4_area_list,
            ch4_point=ch4_point_list,
        )
