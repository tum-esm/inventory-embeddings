from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt

from embeddings.common.log import logger
from embeddings.common.paths import ModelPathsCreator, PlotPaths
from embeddings.dataset.emission_field_transforms import CenterCropTransform
from embeddings.dataset.tno_dataset import TnoDataset
from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection
from embeddings.models.common.metrics import mse
from embeddings.models.vae.vae import VariationalAutoEncoder
from embeddings.plotting.city_emission_field_plot import plot_emission_field_tensor


def _compute_mse_per_city(tno_dataset: TnoDataset) -> dict[str, list[float]]:
    resulting_dict: dict[str, list[float]] = {}
    for c in tno_dataset.city_emission_fields:
        current_mse = resulting_dict.get(c.city_name, [])
        transformed_city = crop_transform(deepcopy(c))
        x = transformed_city.co2_ff_tensor
        x_hat = vae.reconstruct(x)
        current_mse.append(float(mse(x, x_hat)))
        resulting_dict[c.city_name] = current_mse

        if c.city_name == "Frankfurt am Main":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
            plot_emission_field_tensor(emission_field=x, ax=ax1)
            plot_emission_field_tensor(emission_field=x_hat, ax=ax2)
            plt.savefig(PlotPaths.PLOTS / "brussels.png")

    return resulting_dict


if __name__ == "__main__":
    latest_vae = ModelPathsCreator.get_latest_vae_model()
    vae = VariationalAutoEncoder.load_from_checkpoint(checkpoint_path=latest_vae.checkpoint)
    vae.eval()
    crop_transform = CenterCropTransform(TnoDatasetCollection.CROPPED_WIDTH, TnoDatasetCollection.CROPPED_HEIGHT)

    dataset_collection = TnoDatasetCollection()

    datasets = {
        "Test Data": dataset_collection.test_data,
        "Validation Data": dataset_collection.validation_data,
        "Training Data": dataset_collection.training_data,
    }

    for name, data in datasets.items():
        logger.info(f"-------- {name} -------")
        result = _compute_mse_per_city(data)
        for city, mse_values in result.items():
            logger.info(f"\t{city}: {", ".join([str(v) for v in mse_values])}")
        means = []
        for value in result.values():
            means += value
        logger.info(f"Mean: {float(np.array(means).mean())}")
