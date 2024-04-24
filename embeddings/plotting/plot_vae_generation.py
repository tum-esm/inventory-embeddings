import matplotlib.pyplot as plt

from embeddings.common.paths import ModelPaths, PlotPaths
from embeddings.models.vae.vae import VariationalAutoEncoder
from embeddings.plotting.city_emission_field_plot import plot_emission_field_tensor

if __name__ == "__main__":
    first_check_point = next(ModelPaths.VAE_LATEST_CHECKPOINTS.iterdir())
    vae = VariationalAutoEncoder.load_from_checkpoint(checkpoint_path=first_check_point)

    fig, ax = plt.subplots(figsize=(5, 5))

    generated = vae.generate()

    plot_emission_field_tensor(emission_field=generated, ax=ax)

    plt.savefig(PlotPaths.PLOTS / "vae_generated.png")
