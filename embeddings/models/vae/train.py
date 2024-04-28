import argparse

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader

from embeddings.common.log import logger
from embeddings.common.paths import ModelPaths
from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection
from embeddings.models.vae.vae import VariationalAutoEncoder


def train() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    wandb_help = "Argument to toggle weights and biases as logger!"

    parser.add_argument("-wandb", default=False, action="store_true", help=wandb_help)

    args = parser.parse_args()

    ModelPaths.archive_latest_vae_model()

    torch.set_float32_matmul_precision("high")

    tno_dataset = TnoDatasetCollection(deterministic=True)

    train_data = DataLoader(
        dataset=tno_dataset.training_data,
        batch_size=32,
        shuffle=True,
        num_workers=16,
    )

    val_data = DataLoader(
        dataset=tno_dataset.validation_data,
        batch_size=32,
        num_workers=16,
    )

    vae = VariationalAutoEncoder()

    tensorboard_logger = TensorBoardLogger(save_dir=ModelPaths.VAE_LATEST, name="logs", version="tensorboard")
    csv_logger = CSVLogger(save_dir=ModelPaths.VAE_LATEST, name="logs", version="csv")
    loggers = [tensorboard_logger, csv_logger]
    if args.wandb:
        wandb_logger = WandbLogger(project="inventory-embeddings-vae", save_dir=ModelPaths.VAE_LATEST / "logs")
        loggers.append(wandb_logger)

    checkpoint_callback = ModelCheckpoint(
        monitor="validation_loss",
        mode="min",  # model with the smallest validation loss is saved
        dirpath=ModelPaths.VAE_LATEST_CHECKPOINTS,
        filename="{epoch}-{validation_loss:.2f}",
    )

    trainer = Trainer(devices=[0], max_epochs=200, callbacks=[checkpoint_callback], logger=loggers)
    trainer.fit(model=vae, train_dataloaders=train_data, val_dataloaders=val_data)

    logger.info("Training done!")
