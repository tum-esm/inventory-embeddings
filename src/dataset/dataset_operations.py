import numpy as np

from src.dataset.tno_dataset import TnoDataset


def merge(dataset_1: TnoDataset, dataset_2: TnoDataset) -> TnoDataset:
    merged_list = dataset_1.city_emission_fields + dataset_2.city_emission_fields
    return TnoDataset(city_emission_fields=merged_list)


def deterministic_split(dataset: TnoDataset, split: list[float]) -> tuple[TnoDataset, ...]:
    _validate_split(split)
    cities = _get_all_unique_cities_sorted_by_emissions(dataset)
    city_splits = _split(list_=cities, split=split)
    return _create_tno_data_set_splits(dataset=dataset, city_splits=city_splits)


def _get_all_unique_cities_sorted_by_emissions(dataset: TnoDataset) -> list[str]:
    city_to_means_mapping: dict[str, list[float]] = {}
    for field in dataset.city_emission_fields:
        cur_means = city_to_means_mapping.get(field.city_name, [])
        cur_means.append(float(field.co2_ff_area_sources_field.mean()))
        city_to_means_mapping[field.city_name] = cur_means
    city_to_mean_emissions_mapping = {c: np.array(means).mean() for c, means in city_to_means_mapping.items()}

    sorted_by_emissions = dict(sorted(city_to_mean_emissions_mapping.items(), key=lambda item: -item[1]))

    return list(sorted_by_emissions.keys())


def _split(list_: list, split: list[float]) -> list[list]:
    resulting_split = []
    num_elements = len(list_)
    for percentage in split[:-1]:
        num_items_in_split = int(percentage * num_elements) - 1
        step_size = int(num_elements / num_items_in_split)
        start_index = int(step_size / 2) - 1
        current_split = list_[start_index::step_size]
        resulting_split.append(current_split)
        list_ = [item for item in list_ if item not in current_split]
    resulting_split.append(list_)
    return resulting_split


def _create_tno_data_set_splits(dataset: TnoDataset, city_splits: list[list[str]]) -> tuple[TnoDataset, ...]:
    datasets = []
    city_emission_fields = dataset.city_emission_fields
    for split in city_splits:
        sorted_split = sorted(split)
        fields = [c for c in city_emission_fields if c.city_name in sorted_split]
        datasets.append(TnoDataset(fields))
    return tuple(datasets)


def _validate_split(split: list[float]) -> None:
    if sum(split) != 1:
        value_error = "Split does not sum to 1!"
        raise ValueError(value_error)
