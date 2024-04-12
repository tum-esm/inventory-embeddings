from dataclasses import dataclass

from embeddings.common.gnfr_sector_type import GnfrSector


@dataclass
class GhgSource:
    sector: GnfrSector
    co2_ff: float
    co2_bf: float
    ch4: float