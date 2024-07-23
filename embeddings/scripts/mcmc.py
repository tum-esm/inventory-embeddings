import random

import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import MCMC, NUTS
from torch import Tensor

from embeddings.common.gnfr_sector import GnfrSector
from embeddings.common.paths import ModelPathsCreator
from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection
from embeddings.models.vae.vae import VariationalAutoEncoder
from embeddings.plotting.city_emission_field_plot import plot_emission_field_tensor

if __name__ == "__main__":
    # Load the model
    latest_vae = ModelPathsCreator.get_vae_model(model="2048_munich")
    vae = VariationalAutoEncoder.load_from_checkpoint(checkpoint_path=latest_vae.checkpoint)
    latent_dim = vae.latent_dimension
    decoder = vae.decoder


    def generator(z: Tensor) -> Tensor:
        with torch.no_grad():
            generated = decoder(z)
        return generated.squeeze(0)  # 15 by 32 by 32


    def generated_sum(z: Tensor) -> Tensor:
        return torch.sum(generator(z), dim=0)  # 32 by 32


    def generated_sum_vectorized(z: Tensor) -> Tensor:
        return generated_sum(z).reshape(-1) # 1024

    # Load the data
    real_data = False
    if real_data:
        city = "Munich"
        dataset = TnoDatasetCollection().get_case_study_data(city=city, year=2018)
        dataset.disable_temporal_transforms()
        emission_field = dataset[random.randint(0, len(dataset) - 1)].to(vae.device)
    else:
        emission_field = vae.generate().to(vae.device)

    emission_field_sector_sum = torch.sum(emission_field, dim=0) # 32 by 32
    emission_field_sector_sum_vectorized = emission_field_sector_sum.reshape(-1) # 1024

    # generate forward model and observations
    num_measurements = 750
    A = torch.randn(num_measurements, 32 * 32, device=vae.device)
    y_obs = A @ emission_field_sector_sum_vectorized

    # define model for MCMC
    def model() -> None:
        z = pyro.sample(
            "z",
            dist.Normal(
                torch.zeros(latent_dim, device=vae.device),
                torch.ones(latent_dim, device=vae.device),
            ).to_event(1),
        )

        y_sampled = A @ generated_sum_vectorized(z)

        sigma = torch.tensor(0.1, device=vae.device)
        with pyro.plate("data", len(y_obs)):
            pyro.sample("obs", dist.Normal(y_sampled, sigma), obs=y_obs)

    # run MCMC
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=250, warmup_steps=50)
    mcmc.run()

    # get samples
    posterior_samples = mcmc.get_samples()
    z_samples = posterior_samples["z"]

    last_sample = z_samples[-1]
    generated = generator(last_sample).cpu()
    emission_field = emission_field.cpu()

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 2, figsize=(10, 20))

    vmax = 1.1 * float(torch.max(emission_field))

    plot_emission_field_tensor(emission_field=emission_field, ax=ax1[0], vmax=vmax)
    plot_emission_field_tensor(emission_field=emission_field, ax=ax2[0], vmax=vmax, sector=GnfrSector.B)
    plot_emission_field_tensor(emission_field=emission_field, ax=ax3[0], vmax=vmax, sector=GnfrSector.C)
    plot_emission_field_tensor(emission_field=emission_field, ax=ax4[0], vmax=vmax, sector=GnfrSector.F2)

    plot_emission_field_tensor(emission_field=generated, ax=ax1[1], vmax=vmax)
    plot_emission_field_tensor(emission_field=generated, ax=ax2[1], vmax=vmax, sector=GnfrSector.B)
    plot_emission_field_tensor(emission_field=generated, ax=ax3[1], vmax=vmax, sector=GnfrSector.C)
    plot_emission_field_tensor(emission_field=generated, ax=ax4[1], vmax=vmax, sector=GnfrSector.F2)

    plt.show()
