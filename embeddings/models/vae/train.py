import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from embeddings.common.log import logger
from embeddings.common.paths import ModelPaths
from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection
from embeddings.models.vae.vae import VariationalAutoEncoder


def train() -> None:
    ModelPaths.archive_latest_vae_model()

    torch.set_float32_matmul_precision("high")

    tno_dataset = TnoDatasetCollection()

    train_data = DataLoader(
        dataset=tno_dataset.training_data,
        batch_size=32,
        shuffle=True,
        num_workers=16,
    )

    val_data = DataLoader(
        dataset=tno_dataset.validation_data,
        batch_size=128,
        num_workers=16,
    )

    vae = VariationalAutoEncoder()

    train_logger = TensorBoardLogger(save_dir=ModelPaths.VAE_LATEST, name="lightning_logs")

    checkpoint_callback = ModelCheckpoint(
        monitor="validation_loss",
        dirpath=ModelPaths.VAE_LATEST_CHECKPOINTS,
        filename="{epoch}-{validation_loss:.2f}",
    )

    trainer = Trainer(devices=[0], max_epochs=100, callbacks=[checkpoint_callback], logger=train_logger)
    trainer.fit(model=vae, train_dataloaders=train_data, val_dataloaders=val_data)

    logger.info("Training done!")
