import random

import matplotlib.pyplot as plt

from embeddings.common.gnfr_sector import GnfrSector
from embeddings.common.paths import PlotPaths
from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection

if __name__ == "__main__":
    dataset_collection = TnoDatasetCollection()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    emission_field = random.choice(dataset_collection.test_data)
    emission_field.plot(ax=ax1)
    emission_field.plot(ax=ax2, sector=GnfrSector.F1)

    plt.savefig(PlotPaths.PLOTS / "city_plot.png")
