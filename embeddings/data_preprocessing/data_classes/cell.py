from dataclasses import dataclass
from typing import Self

from embeddings.data_preprocessing.data_classes.ghg_source import GhgSource


@dataclass
class Cell:
    lon: float
    lat: float
    co2_ff: list[float]
    co2_bf: list[float]
    ch4: list[float]

    @classmethod
    def from_ghg_sources(
        cls,
        lon: float,
        lat: float,
        sources: list[GhgSource],
    ) -> Self:
        co2_ff_list = [0.0 for _ in range(15)]
        co2_bf_list = [0.0 for _ in range(15)]
        ch4_list = [0.0 for _ in range(15)]
        for source in sources:
            co2_ff_list[source.sector.to_index()] = source.co2_ff
            co2_bf_list[source.sector.to_index()] = source.co2_bf
            ch4_list[source.sector.to_index()] = source.ch4
        return Cell(lon, lat, co2_ff_list, co2_bf_list, ch4_list)

    @property
    def co2_ff_str(self) -> str:
        return ",".join([str(val) for val in self.co2_ff])

    @property
    def co2_bf_str(self) -> str:
        return ",".join([str(val) for val in self.co2_bf])

    @property
    def ch4_str(self) -> str:
        return ",".join([str(val) for val in self.ch4])
