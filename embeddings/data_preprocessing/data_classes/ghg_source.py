from dataclasses import dataclass

from embeddings.common.gnfr_sector_type import GnfrSectorType


@dataclass
class GhgSource:
    sector: GnfrSectorType
    co2_ff: float
    co2_bf: float
    ch4: float
