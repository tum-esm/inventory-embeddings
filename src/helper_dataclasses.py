from __future__ import annotations

from dataclasses import dataclass

from src.gnfr_sector_type import GnfrSectorType


@dataclass
class GhgSource:
    sector: GnfrSectorType
    co2_ff: float
    co2_bf: float
    ch4: float


@dataclass
class Cell:
    lon: float
    lat: float
    co2_ff: list[float]

    @staticmethod
    def from_ghg_sources(
        lon: float,
        lat: float,
        sources: list[GhgSource],
    ) -> Cell:
        co2_ff_list = [0.0 for _ in range(15)]
        for source in sources:
            co2_ff_list[source.sector.to_index()] = source.co2_ff
        return Cell(lon, lat, co2_ff_list)

    def is_left_of(self, other: Cell) -> bool:
        return self.lon < other.lon

    def is_right_of(self, other: Cell) -> bool:
        return self.lon > other.lon

    def is_below(self, other: Cell) -> bool:
        return self.lat < other.lat

    def is_above(self, other: Cell) -> bool:
        return self.lat > other.lat
