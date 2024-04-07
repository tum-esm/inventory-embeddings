from dataclasses import dataclass

from src.data_set.convert.sector import Sector


@dataclass
class Cell:
    long: float
    lat: float
    iso_codes: list[str]
    sectors: list[Sector]

    def append_sector(self, sector: Sector) -> None:
        existing_sectors = [s for s in self.sectors if s.gnfr_type == sector.gnfr_type]
        new_sector = Sector.empty(sector.gnfr_type)
        if len(existing_sectors) > 0:
            new_sector = existing_sectors[0]
        new_sector = new_sector + sector
        self.sectors.append(new_sector)

    def append_iso_code(self, iso_code: str) -> None:
        if iso_code not in self.iso_codes:
            self.iso_codes.append(iso_code)

    def _value_as_list(self, value: str) -> list[str]:
        result = ["0.0" for _ in range(15)]
        for sector in self.sectors:
            result[sector.gnfr_type.value - 1] = str(getattr(sector, value))
        return result

    def to_row(self) -> list[str | float]:
        return [
            self.long,
            self.lat,
            ",".join(self.iso_codes),
            ",".join(self._value_as_list("co2_ff")),
            ",".join(self._value_as_list("co2_bf")),
            ",".join(self._value_as_list("ch4")),
        ]
