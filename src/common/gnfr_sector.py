from enum import Enum
from typing import Self

NUM_GNFR_SECTORS = 15


class GnfrSector(Enum):
    A = 1
    B = 2
    C = 3
    D = 4
    E = 5
    F1 = 6
    F2 = 7
    F3 = 8
    F4 = 9
    G = 10
    H = 11
    I = 12
    J = 13
    K = 14
    L = 15

    def to_index(self) -> int:
        return self.value - 1

    @classmethod
    def from_index(cls, index: int) -> Self:
        return cls(value=index + 1)

    @classmethod
    def from_str(cls, sector: str) -> Self:
        return cls[sector]
