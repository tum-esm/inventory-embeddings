import random

import matplotlib.pyplot as plt

from embeddings.common.gnfr_sector import GnfrSector
from embeddings.common.paths import PlotPaths
from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection
from embeddings.plotting.city_emission_field_plot import plot_emission_field

if __name__ == "__main__":
    dataset_collection = TnoDatasetCollection()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    data = dataset_collection.training_data

    emission_field = data.get_city_emission_field(random.randint(0, len(data) - 1))
    plot_emission_field(emission_field=emission_field, ax=ax1)
    plot_emission_field(emission_field=emission_field, ax=ax2, sector=GnfrSector.F1)

    plt.savefig(PlotPaths.PLOTS / "city_plot.png")
