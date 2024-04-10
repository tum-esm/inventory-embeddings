from src.filter_cities import filter_cities
from src.paths import TNO_HIGH_RES_2015

if __name__ == "__main__":
    tno_2015 = TNO_HIGH_RES_2015 / "TNO_GHGco_2015_highres_v1_1.csv"

    filter_cities(
        tno_data_csv=tno_2015,
        out_csv=TNO_HIGH_RES_2015 / "TNO_GHGco_2015_highres_v1_1_by_city.csv",
    )
