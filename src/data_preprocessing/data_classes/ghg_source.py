from dataclasses import dataclass

from src.common.gnfr_sector import GnfrSector


@dataclass
class GhgSource:
    sector: GnfrSector
    co2_ff: float
    co2_bf: float
    ch4: float


@dataclass
class GhgPointSource(GhgSource):
    lat: float
    lon: float
