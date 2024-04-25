from embeddings.common.paths import TnoPaths
from embeddings.dataset.dataset_split import deterministic_split, random_split
from embeddings.dataset.emission_field_transforms import (
    CenterCropTransform,
    RandomCropTransform,
    RandomHorizontalFlipTransform,
    RandomVerticalFlipTransform,
)
from embeddings.dataset.tno_dataset import TnoDataset


class TnoDatasetCollection:
    def __init__(self, deterministic: bool = False) -> None:
        tno_2015 = TnoDataset.from_csv(TnoPaths.BY_CITY_2015_CSV)

        _, rest = deterministic_split(tno_2015, split=[0.9, 0.1])

        val_split = 0.15

        if deterministic:
            self._val, self._train = deterministic_split(rest, split=[val_split, 1 - val_split])
        else:
            self._val, self._train = random_split(rest, split=[val_split, 1 - val_split])

        self._add_sampling_transforms()

    def _add_sampling_transforms(self) -> None:
        self._train.add_sampling_transform(RandomCropTransform(width=32, height=32))
        self._train.add_sampling_transform(RandomHorizontalFlipTransform())
        self._train.add_sampling_transform(RandomVerticalFlipTransform())

        self._val.add_sampling_transform(CenterCropTransform(width=32, height=32))

    @property
    def validation_data(self) -> TnoDataset:
        return self._val

    @property
    def training_data(self) -> TnoDataset:
        return self._train
