import random

import matplotlib.pyplot as plt
import polars as pl

from embeddings.common.gnfr_sector import GnfrSector
from embeddings.common.paths import PlotPaths, TnoPaths
from embeddings.dataset.city_emission_grid import CityEmissionField

if __name__ == "__main__":
    tno_data = pl.read_csv(TnoPaths.TNO_BY_CITY_2015_CSV, separator=";")
    cites = list(tno_data["City"].unique())

    city_data = tno_data.filter(pl.col("City") == random.choice(cites))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    emission_field = CityEmissionField(city_data=city_data)
    emission_field.plot(ax=ax1, sector=None)
    emission_field.plot(ax=ax2, sector=GnfrSector.F1)

    plt.savefig(PlotPaths.PLOTS / "city_plot.png")
