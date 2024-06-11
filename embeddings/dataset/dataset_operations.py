import random

from embeddings.dataset.tno_dataset import TnoDataset


def merge(dataset_1: TnoDataset, dataset_2: TnoDataset) -> TnoDataset:
    merged_list = dataset_1.city_emission_fields + dataset_2.city_emission_fields
    return TnoDataset(city_emission_fields=merged_list)


def random_split(dataset: TnoDataset, split: list[float]) -> tuple[TnoDataset, ...]:
    _validate_split(split)
    cities = _get_all_unique_cities(dataset)
    random.shuffle(cities)
    city_splits = _split(list_=cities, split=split)
    return _create_tno_data_set_splits(dataset=dataset, city_splits=city_splits)


def deterministic_split(dataset: TnoDataset, split: list[float]) -> tuple[TnoDataset, ...]:
    _validate_split(split)
    cities = _get_all_unique_cities(dataset)
    city_splits = _split(list_=cities, split=split)
    return _create_tno_data_set_splits(dataset=dataset, city_splits=city_splits)


def _get_all_unique_cities(dataset: TnoDataset) -> list[str]:
    cities = [field.city_name for field in dataset.city_emission_fields]
    return sorted(dict.fromkeys(cities))


def _split(list_: list, split: list[float]) -> list[list]:
    resulting_split = []
    num_elements = len(list_)
    start_index = 0
    for part in split[:-1]:
        end_index = start_index + int(part * num_elements)
        resulting_split.append(list_[start_index:end_index])
        start_index = end_index
    resulting_split.append(list_[start_index:])
    return resulting_split


def _create_tno_data_set_splits(dataset: TnoDataset, city_splits: list[list[str]]) -> tuple[TnoDataset, ...]:
    datasets = []
    city_emission_fields = dataset.city_emission_fields
    for split in city_splits:
        fields = [c for c in city_emission_fields if c.city_name in split]
        datasets.append(TnoDataset(fields))
    return tuple(datasets)


def _validate_split(split: list[float]) -> None:
    if sum(split) != 1:
        value_error = "Split does not sum to 1!"
        raise ValueError(value_error)
