import random

import matplotlib.pyplot as plt
import torch

from embeddings.common.gnfr_sector import GnfrSector
from embeddings.common.paths import ModelPaths, PlotPaths
from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection
from embeddings.models.vae.vae import VariationalAutoEncoder
from embeddings.plotting.city_emission_field_plot import plot_emission_field_tensor

if __name__ == "__main__":
    first_check_point = next(ModelPaths.VAE_LATEST_CHECKPOINTS.iterdir())
    vae = VariationalAutoEncoder.load_from_checkpoint(checkpoint_path=first_check_point)

    dataset_collection = TnoDatasetCollection(deterministic=True)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))

    data = dataset_collection.training_data

    emission_field = data[random.randint(0, len(data) - 1)]

    reconstructed = vae.reconstruct(emission_field)

    vmax = float(torch.max(emission_field))

    plot_emission_field_tensor(emission_field=emission_field, ax=ax1, vmax=vmax)
    plot_emission_field_tensor(emission_field=emission_field, ax=ax2, sector=GnfrSector.F1, vmax=vmax)

    plot_emission_field_tensor(emission_field=reconstructed, ax=ax3, vmax=vmax)
    plot_emission_field_tensor(emission_field=reconstructed, ax=ax4, sector=GnfrSector.F1, vmax=vmax)

    plt.savefig(PlotPaths.PLOTS / "vae_reconstructed.png")
