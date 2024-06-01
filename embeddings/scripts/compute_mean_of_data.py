import numpy as np

from embeddings.common.log import logger
from embeddings.dataset.tno_dataset import TnoDataset
from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection


def _compute_mean(tno_dataset: TnoDataset) -> float:
    means = np.zeros(len(tno_dataset.city_emission_fields))
    for i, city in enumerate(tno_dataset.city_emission_fields):
        means[i] = city.co2_ff_field.mean()
    return float(means.mean())


if __name__ == "__main__":
    dataset_collection = TnoDatasetCollection()

    logger.info(f"Mean of training data:\t\t{_compute_mean(dataset_collection.training_data)}")
    logger.info(f"Mean of validation data:\t{_compute_mean(dataset_collection.validation_data)}")
    logger.info(f"Mean of test data:\t\t\t{_compute_mean(dataset_collection.test_data)}")
