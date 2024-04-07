import csv
from pathlib import Path

import pandas as pd
from alive_progress import alive_bar

from src.data_set.convert.cell import Cell
from src.data_set.convert.gnfr_sector_type import GnfrSectorType
from src.data_set.convert.sector import Sector

_DATA = Path("data")


def convert_cams_v4_dataset(file_name: str) -> None:
    data: pd.DataFrame = pd.read_csv(_DATA / file_name, sep=";")

    groups = data.groupby(["Lon_rounded", "Lat_rounded"]).groups

    base_file_name = ".".join(file_name.split(".")[:-1])
    preprocessed = _DATA / "converted"
    preprocessed.mkdir(exist_ok=True)
    out_path = preprocessed / f"{base_file_name}_converted.csv"

    with alive_bar(total=len(data)) as bar, out_path.open("w") as file:
        csv_writer = csv.writer(file, delimiter=";")
        csv_writer.writerow(["Long", "Lat", "ISO3_Codes", "CO2_ff_sectors", "CO2_bf_sectors", "CH4_sectors"])

        for location, rows in groups.items():
            grid = Cell(long=location[0], lat=location[1], iso_codes=[], sectors=[])

            for row in rows:
                sector_data = data.iloc[row]
                grid.append_iso_code(sector_data["ISO3"])
                grid.append_sector(
                    sector=Sector(
                        gnfr_type=GnfrSectorType[sector_data["GNFR_Sector"]],
                        ch4=sector_data["CH4"],
                        co2_bf=sector_data["CO2_bf"],
                        co2_ff=sector_data["CO2_ff"],
                    ),
                )
                bar()
            csv_writer.writerow(grid.to_row())
