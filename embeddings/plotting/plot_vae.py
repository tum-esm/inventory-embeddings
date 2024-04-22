import random

import matplotlib.pyplot as plt
import torch

from embeddings.common.paths import ModelPaths, PlotPaths
from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection
from embeddings.models.device import device
from embeddings.models.vae.vae import VariationalAutoEncoder
from embeddings.plotting.city_emission_field_plot import plot_emission_field_tensor

if __name__ == "__main__":
    vae = VariationalAutoEncoder().to(device)
    vae.load_state_dict(torch.load(ModelPaths.VAE_LATEST_MODEL))

    dataset_collection = TnoDatasetCollection()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    data = dataset_collection.training_data

    emission_field = data.get_city_emission_field(random.randint(0, len(data) - 1))

    input_ = emission_field.co2_ff_tensor.unsqueeze(0).to(device)
    reconstructed, _, _ = vae(input_)
    output = reconstructed.squeeze(0).detach().cpu()

    plot_emission_field_tensor(emission_field=emission_field.co2_ff_tensor, ax=ax1)
    plot_emission_field_tensor(emission_field=output, ax=ax2)

    plt.savefig(PlotPaths.PLOTS / "city_plot.png")
