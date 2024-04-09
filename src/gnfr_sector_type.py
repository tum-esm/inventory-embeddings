from __future__ import annotations

from enum import Enum


class GnfrSectorType(Enum):
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

    @staticmethod
    def from_index(index: int) -> GnfrSectorType:
        return GnfrSectorType(value=index + 1)

    @staticmethod
    def from_str(sector: str) -> GnfrSectorType:
        return GnfrSectorType[sector]
