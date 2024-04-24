import random

import matplotlib.pyplot as plt

from embeddings.common.paths import ModelPaths, PlotPaths
from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection
from embeddings.models.vae.vae import VariationalAutoEncoder
from embeddings.plotting.city_emission_field_plot import plot_emission_field_tensor

if __name__ == "__main__":
    first_check_point = next(ModelPaths.VAE_LATEST_CHECKPOINTS.iterdir())
    vae = VariationalAutoEncoder.load_from_checkpoint(checkpoint_path=first_check_point)

    dataset_collection = TnoDatasetCollection(deterministic=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    data = dataset_collection.training_data

    emission_field = data.get_city_emission_field(random.randint(0, len(data) - 1))

    reconstructed = vae.reconstruct(emission_field.co2_ff_tensor)

    plot_emission_field_tensor(emission_field=emission_field.co2_ff_tensor, ax=ax1)
    plot_emission_field_tensor(emission_field=reconstructed, ax=ax2)

    plt.savefig(PlotPaths.PLOTS / "city_plot.png")
