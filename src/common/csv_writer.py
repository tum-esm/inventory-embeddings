import csv
from pathlib import Path


class CsvWriter:
    def __init__(self, path: Path) -> None:
        self._path = path

    def write_header(self, *args: str) -> None:
        with self._path.open("w") as file:
            csv_reader = csv.writer(file)
            csv_reader.writerow(args)

    def write_row(self, *args: str | float | None) -> None:
        with self._path.open("a") as file:
            csv_reader = csv.writer(file)
            csv_reader.writerow(args)

    def write_rows(self, rows: list[list[str | float]]) -> None:
        with self._path.open("a") as file:
            csv_reader = csv.writer(file)
            csv_reader.writerows(rows)
