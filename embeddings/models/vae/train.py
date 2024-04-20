from torch.utils.data import DataLoader
from torchsummary import summary

from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection
from embeddings.models.device import device
from embeddings.models.vae.vae import VariationalAutoEncoder


def train() -> None:
    print(device)
    vae = VariationalAutoEncoder().to(device)
    summary(model=vae, input_size=(15, 32, 32), device=device)

    tno_dataset = TnoDatasetCollection()

    batch_size = 32

    train_data_loader = DataLoader(
        dataset=tno_dataset.training_data,
        batch_size=batch_size,
        shuffle=True,
    )

    for data in iter(train_data_loader):
        print(data.shape)
