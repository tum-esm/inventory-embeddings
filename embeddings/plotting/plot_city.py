import random

import matplotlib.pyplot as plt

from embeddings.common.gnfr_sector import GnfrSector
from embeddings.common.paths import PlotPaths, TnoPaths
from embeddings.dataset.dataset_split import random_split
from embeddings.dataset.tno_dataset import TnoDataset

if __name__ == "__main__":
    dataset = TnoDataset.from_csv(TnoPaths.BY_CITY_2015_CSV)
    [test, validation, training] = random_split(dataset, split=[0.15, 0.15, 0.7])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    emission_field = random.choice(test)
    emission_field.plot(ax=ax1)
    emission_field.plot(ax=ax2, sector=GnfrSector.F1)

    plt.savefig(PlotPaths.PLOTS / "city_plot.png")
