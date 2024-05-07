import random

import matplotlib.pyplot as plt

from embeddings.common.gnfr_sector import GnfrSector
from embeddings.common.paths import PlotPaths
from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection
from embeddings.plotting.city_emission_field_plot import plot_emission_field_sector

if __name__ == "__main__":
    dataset_collection = TnoDatasetCollection(deterministic=True)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))

    data = dataset_collection.training_data

    random_index = random.randint(0, len(data) - 1)

    emission_field = data.get_city_emission_field(random_index)

    transformed_emission_field = data.get_city_emission_field(random_index, apply_sampling_transforms=True)

    plot_emission_field_sector(emission_field=emission_field, ax=ax1)
    plot_emission_field_sector(emission_field=emission_field, ax=ax2, sector=GnfrSector.F1)

    plot_emission_field_sector(emission_field=transformed_emission_field, ax=ax3)
    plot_emission_field_sector(emission_field=transformed_emission_field, ax=ax4, sector=GnfrSector.F1)

    plt.savefig(PlotPaths.PLOTS / "city_plot.png")
