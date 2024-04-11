import matplotlib.pyplot as plt
import polars as pl
from mpl_toolkits.basemap import Basemap

from embeddings.common.paths import PLOTS, TnoPaths

if __name__ == "__main__":
    tno_data = pl.read_csv(TnoPaths.TNO_BY_CITY_2015_CSV, separator=";")
    city_data = tno_data.filter(pl.col("City") == "Copenhagen")

    min_lat = city_data["lat"].min()
    max_lat = city_data["lat"].max()
    min_lon = city_data["lon"].min()
    max_lon = city_data["lon"].max()

    fig = plt.figure(figsize=(8, 8))

    # Map Look
    m = Basemap(
        projection="merc",
        llcrnrlat=min_lat,
        urcrnrlat=max_lat,
        llcrnrlon=min_lon,
        urcrnrlon=max_lon,
        resolution="h",
    )
    m.drawcoastlines()
    m.fillcontinents(color="coral", lake_color="aqua")
    m.drawmapboundary(fill_color="aqua")

    plt.savefig(PLOTS / "plot.png")
