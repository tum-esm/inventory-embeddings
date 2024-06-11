from embeddings.common.log import logger
from embeddings.common.paths import TnoPaths
from embeddings.dataset.dataset_operations import deterministic_split, merge, random_split
from embeddings.dataset.emission_field_transforms import (
    CenterCropTransform,
    RandomCropTransform,
    RandomHorizontalFlipTransform,
    RandomRotationTransform,
    RandomVerticalFlipTransform,
)
from embeddings.dataset.tno_dataset import TnoDataset

CITIES_TO_REMOVE = [
    "Bratislava",
]

CITIES_FOR_CASE_STUDY = [
    "Munich",
    "Zürich",
    "Paris",
]


class TnoDatasetCollection:
    CROPPED_WIDTH = 32
    CROPPED_HEIGHT = 32

    def __init__(self, test_split: float = 0.15, val_split: float = 0.15, random: bool = False) -> None:
        tno_2015 = TnoDataset.from_csv(TnoPaths.BY_CITY_2015_CSV)
        tno_2018 = TnoDataset.from_csv(TnoPaths.BY_CITY_2018_CSV)
        self._complete_tno = merge(tno_2015, tno_2018)

        self._remove_excluded_cities()
        self._build_case_study_datasets()

        self._test, rest = deterministic_split(self._complete_tno, split=[test_split, 1 - test_split])

        rest_val_split = val_split / (1 - test_split)

        if random:
            self._val, self._train = random_split(rest, split=[rest_val_split, 1 - rest_val_split])
        else:
            self._val, self._train = deterministic_split(rest, split=[rest_val_split, 1 - rest_val_split])

        logger.info(f"Test Set has {self._test.num_unique_cities} unique cites!\n\t{self._test!s}")
        logger.info(f"Validation Set has {self._val.num_unique_cities} unique cites!\n\t{self._val!s}")
        logger.info(f"Training Set has {self._train.num_unique_cities} unique cites!\n\t{self._train!s}")

        self._add_sampling_transforms()

    def _remove_excluded_cities(self) -> None:
        for city in CITIES_TO_REMOVE:
            self._complete_tno.remove_city_with_name(name=city)

    def _build_case_study_datasets(self) -> None:
        self._case_study_datasets = {}
        for city in CITIES_FOR_CASE_STUDY:
            emission_fields = self._complete_tno.remove_city_with_name(name=city)
            self._case_study_datasets[city] = TnoDataset(city_emission_fields=emission_fields)
            center_crop_transform = CenterCropTransform(width=self.CROPPED_WIDTH, height=self.CROPPED_HEIGHT)
            self._case_study_datasets[city].add_sampling_transform(center_crop_transform)

    def _add_sampling_transforms(self) -> None:
        self._train.add_sampling_transform(RandomCropTransform(width=self.CROPPED_WIDTH, height=self.CROPPED_HEIGHT))
        self._train.add_sampling_transform(RandomHorizontalFlipTransform())
        self._train.add_sampling_transform(RandomVerticalFlipTransform())
        self._train.add_sampling_transform(RandomRotationTransform())

        self._val.add_sampling_transform(CenterCropTransform(width=self.CROPPED_WIDTH, height=self.CROPPED_HEIGHT))

        self._test.add_sampling_transform(CenterCropTransform(width=self.CROPPED_WIDTH, height=self.CROPPED_HEIGHT))

    @property
    def test_data(self) -> TnoDataset:
        return self._test

    @property
    def validation_data(self) -> TnoDataset:
        return self._val

    @property
    def training_data(self) -> TnoDataset:
        return self._train

    @property
    def complete_data(self) -> TnoDataset:
        return self._complete_tno

    def get_case_study_data(self, city: str) -> TnoDataset:
        return self._case_study_datasets[city]
