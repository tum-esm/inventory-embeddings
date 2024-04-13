import random

import matplotlib.pyplot as plt
import polars as pl

from embeddings.common.gnfr_sector import GnfrSector
from embeddings.common.paths import PlotPaths, TnoPaths
from embeddings.dataset.city_emission_field import CityEmissionField
from embeddings.dataset.emission_field_transforms import CropTransform, HourTransform

if __name__ == "__main__":
    tno_data = pl.read_csv(TnoPaths.BY_CITY_2015_CSV, separator=";")
    cites = list(tno_data["City"].unique())

    crop_transform = CropTransform(center_offset_x=-10, center_offset_y=10, width=32, height=32)
    hour_transform = HourTransform(hour=1)

    city = random.choice(cites)
    city_data = tno_data.filter(pl.col("City") == city)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    emission_field = CityEmissionField(city_data=city_data)
    emission_field = hour_transform(crop_transform(emission_field))
    emission_field.plot(ax=ax1, sector=None)
    emission_field.plot(ax=ax2, sector=GnfrSector.F1)

    plt.savefig(PlotPaths.PLOTS / "city_plot.png")
