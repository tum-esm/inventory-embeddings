from dataclasses import dataclass
from typing import Self

from embeddings.data_preprocessing.data_classes.ghg_source import GhgSource
from embeddings.data_preprocessing.tno_constants import TNO_LAT_STEP, TNO_LON_STEP


@dataclass
class Cell:
    lat: float
    lon: float
    x: int
    y: int
    co2_ff: list[float]
    co2_bf: list[float]
    ch4: list[float]

    @property
    def co2_ff_str(self) -> str:
        return ",".join([str(val) for val in self.co2_ff])

    @property
    def co2_bf_str(self) -> str:
        return ",".join([str(val) for val in self.co2_bf])

    @property
    def ch4_str(self) -> str:
        return ",".join([str(val) for val in self.ch4])


class CellBuilder:
    def __init__(self) -> None:
        self._lat = None
        self._lon = None
        self._lon_or = None
        self._lat_or = None
        self._ghg_sources = None

    def with_coordinates(self, lat: float, lon: float) -> Self:
        self._lat = lat
        self._lon = lon
        return self

    def with_coordinates_origin(self, lon: float, lat: float) -> Self:
        self._lon_or = lon
        self._lat_or = lat
        return self

    def with_ghg_sources(self, sources: list[GhgSource]) -> Self:
        self._ghg_sources = sources
        return self

    def _compute_pixel_coordinates(self) -> tuple[int, int]:
        x = round((self._lon - self._lon_or) / TNO_LON_STEP)
        y = round((self._lat_or - self._lat) / TNO_LAT_STEP)
        return x, y

    def build(self) -> Cell:
        co2_ff_list = [0.0 for _ in range(15)]
        co2_bf_list = [0.0 for _ in range(15)]
        ch4_list = [0.0 for _ in range(15)]
        for source in self._ghg_sources:
            co2_ff_list[source.sector.to_index()] = source.co2_ff
            co2_bf_list[source.sector.to_index()] = source.co2_bf
            ch4_list[source.sector.to_index()] = source.ch4
        x, y = self._compute_pixel_coordinates()
        return Cell(
            lat=self._lat,
            lon=self._lon,
            x=x,
            y=y,
            co2_ff=co2_ff_list,
            co2_bf=co2_bf_list,
            ch4=ch4_list,
        )
