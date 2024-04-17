from embeddings.common.paths import TnoPaths
from embeddings.dataset.dataset_split import deterministic_split, random_split
from embeddings.dataset.tno_dataset import TnoDataset


class TnoDatasetCollection:
    def __init__(self) -> None:
        tno_2015 = TnoDataset.from_csv(TnoPaths.BY_CITY_2015_CSV)

        test_split = 0.15
        val_split = 0.15

        val_part = val_split / (1 - test_split)

        self._test, train_val = deterministic_split(tno_2015, split=[test_split, 1 - test_split])
        self._val, self._train = random_split(train_val, split=[val_part, 1 - val_part])

    @property
    def test_data(self) -> TnoDataset:
        return self._test

    @property
    def validation_data(self) -> TnoDataset:
        return self._val

    @property
    def training_data(self) -> TnoDataset:
        return self._train
