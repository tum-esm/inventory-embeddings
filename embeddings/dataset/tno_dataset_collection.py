from embeddings.common.paths import TnoPaths
from embeddings.dataset.dataset_split import deterministic_split, random_split
from embeddings.dataset.tno_dataset import TnoDataset


class TnoDatasetCollection:
    def __init__(self) -> None:
        tno_2015 = TnoDataset.from_csv(TnoPaths.BY_CITY_2015_CSV)

        _, rest = deterministic_split(tno_2015, split=[0.9, 0.1])

        val_split = 0.15

        self._val, self._train = random_split(rest, split=[val_split, 1 - val_split])

    @property
    def validation_data(self) -> TnoDataset:
        return self._val

    @property
    def training_data(self) -> TnoDataset:
        return self._train
