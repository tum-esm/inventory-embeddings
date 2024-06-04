import matplotlib.pyplot as plt

from embeddings.common.paths import ModelPaths
from embeddings.models.vae.vae import VariationalAutoEncoder
from embeddings.plotting.city_emission_field_plot import plot_emission_field_tensor

if __name__ == "__main__":
    latest_vae = ModelPaths.get_latest_vae_model()
    vae = VariationalAutoEncoder.load_from_checkpoint(checkpoint_path=latest_vae.checkpoint)

    fig, ax = plt.subplots(figsize=(5, 5))

    generated = vae.generate()

    plot_emission_field_tensor(emission_field=generated, ax=ax)

    latest_vae.plots.mkdir(exist_ok=True)
    plt.savefig(latest_vae.plots / "generated.png")
