from embeddings.common.log import logger
from embeddings.common.paths import TnoPaths
from embeddings.dataset.dataset_split import deterministic_split, random_split
from embeddings.dataset.emission_field_transforms import (
    CenterCropTransform,
    RandomCropTransform,
    RandomHorizontalFlipTransform,
    RandomRotationTransform,
    RandomVerticalFlipTransform,
)
from embeddings.dataset.tno_dataset import TnoDataset


class TnoDatasetCollection:
    CROPPED_WIDTH = 32
    CROPPED_HEIGHT = 32

    def __init__(self) -> None:
        tno_2015 = TnoDataset.from_csv(TnoPaths.BY_CITY_2015_CSV)

        val_split = 0.15
        test_split = 0.15

        self._test, rest = deterministic_split(tno_2015, split=[test_split, 1 - test_split])

        rest_val_split = val_split / (1 - test_split)

        self._val, self._train = random_split(rest, split=[rest_val_split, 1 - rest_val_split])

        logger.info(f"Test Set has {len(self._test.city_emission_fields)} cites!")
        logger.info(f"Validation Set has {len(self._val.city_emission_fields)} cites!")
        logger.info(f"Training Set has {len(self._train.city_emission_fields)} cites!")

        self._add_sampling_transforms()

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
