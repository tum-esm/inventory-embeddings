from torch.utils.data import DataLoader

from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection


def train() -> None:
    tno_dataset = TnoDatasetCollection()

    batch_size = 32

    train_data_loader = DataLoader(
        dataset=tno_dataset.training_data,
        batch_size=batch_size,
        shuffle=True,
    )

    for data in iter(train_data_loader):
        print(data.shape)
