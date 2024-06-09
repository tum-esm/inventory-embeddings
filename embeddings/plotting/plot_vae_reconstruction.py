import random

import matplotlib.pyplot as plt
import torch

from embeddings.common.gnfr_sector import GnfrSector
from embeddings.common.log import logger
from embeddings.common.paths import ModelPathsCreator
from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection
from embeddings.models.common.metrics import mse, ssim
from embeddings.models.vae.vae import VariationalAutoEncoder
from embeddings.plotting.city_emission_field_plot import plot_emission_field_tensor

if __name__ == "__main__":
    latest_vae = ModelPathsCreator.get_latest_vae_model()
    vae = VariationalAutoEncoder.load_from_checkpoint(checkpoint_path=latest_vae.checkpoint)

    dataset_collection = TnoDatasetCollection()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))

    data = dataset_collection.test_data

    emission_field = data[random.randint(0, len(data) - 1)]

    reconstructed = vae.reconstruct(emission_field)

    sector = GnfrSector.F2

    vmax = 1.1 * float(torch.max(emission_field))

    plot_emission_field_tensor(emission_field=emission_field, ax=ax1, vmax=vmax)
    plot_emission_field_tensor(emission_field=emission_field, ax=ax2, sector=sector, vmax=vmax)

    plot_emission_field_tensor(emission_field=reconstructed, ax=ax3, vmax=vmax)
    plot_emission_field_tensor(emission_field=reconstructed, ax=ax4, sector=sector, vmax=vmax)

    latest_vae.plots.mkdir(exist_ok=True)
    plt.savefig(latest_vae.plots / "reconstructed.png")

    logger.info(f"MSE: {mse(x=emission_field, x_hat=reconstructed)}")
    logger.info(f"SSIM: {ssim(x=emission_field, x_hat=reconstructed)}")
