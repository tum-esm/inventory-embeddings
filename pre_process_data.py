from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import pandas as pd

from src.gnfr_sector_type import GnfrSectorType
from src.helper_dataclasses import Cell, GhgSource

DATA = Path("data")
TNO_HIGH_RES = DATA / "TNO-GHGco-1km"
TNO_HIGH_RES_2015 = TNO_HIGH_RES / "TNO_highres_2015"

LAT_STEP = 1 / 120  # "vertical"; the bigger, the more north
LONG_STEP = 1 / 60  # "horizontal"; the bigger, the more east

MUNICH_LAT = 48.13743
MUNICH_LONG = 11.57549


@contextlib.contextmanager
def time_it(text: str) -> Generator:
    start_time = time.time()
    yield
    print(f"{text} took {time.time() - start_time}s")


@dataclass
class Grid:
    rows: list[list[Cell]]


if __name__ == "__main__":
    tno_2015 = TNO_HIGH_RES_2015 / "TNO_GHGco_2015_highres_v1_1.csv"

    with time_it("Reading csv"):
        tno_2015_data = pd.read_csv(tno_2015, sep=";")

    cells = []

    with time_it("Filtering for location"):
        area_source_type = tno_2015_data[
            (tno_2015_data["Lat"] >= MUNICH_LAT - 30.5 * LAT_STEP)
            & (tno_2015_data["Lat"] <= MUNICH_LAT + 30.5 * LAT_STEP)
            & (tno_2015_data["Lon"] >= MUNICH_LONG - 30.5 * LONG_STEP)
            & (tno_2015_data["Lon"] <= MUNICH_LONG + 30.5 * LONG_STEP)
            & (tno_2015_data["SourceType"] == "A")
        ]
        lat_set = set()
        lon_set = set()
        cells_data = area_source_type.groupby(["Lon", "Lat"]).groups
        for (lon, lat), rows in cells_data.items():
            sources = []
            for row in rows:
                source = tno_2015_data.iloc[row]
                sources.append(
                    GhgSource(
                        sector=GnfrSectorType.from_str(source["GNFR_Sector"]),
                        co2_ff=source["CO2_ff"],
                        co2_bf=source["CO2_bf"],
                        ch4=source["CH4"],
                    ),
                )
            cells.append(Cell.from_ghg_sources(lon, lat, sources))
