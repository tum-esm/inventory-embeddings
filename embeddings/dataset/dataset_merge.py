from embeddings.dataset.tno_dataset import TnoDataset


def merge(dataset_1: TnoDataset, dataset_2: TnoDataset) -> TnoDataset:
    merged_list = dataset_1.city_emission_fields + dataset_2.city_emission_fields
    return TnoDataset(city_emission_fields=merged_list)
