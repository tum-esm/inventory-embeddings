from dataclasses import dataclass

from src.data_set.convert.gnfr_sector_type import GnfrSectorType


@dataclass
class Sector:
    gnfr_type: GnfrSectorType
    co2_ff: float
    co2_bf: float
    ch4: float

    def __add__(self, other: "Sector") -> "Sector":
        return Sector(
            gnfr_type=self.gnfr_type,
            co2_ff=self.co2_ff + other.co2_ff,
            co2_bf=self.co2_bf + other.co2_bf,
            ch4=self.ch4 + other.ch4,
        )

    @staticmethod
    def empty(sector_type: GnfrSectorType) -> "Sector":
        return Sector(
            gnfr_type=sector_type,
            ch4=0,
            co2_bf=0,
            co2_ff=0,
        )
