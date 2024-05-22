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

WANDB_PROJECT_NAME = "inventory-embeddings-vae-ablations-more-data"


def train() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    epochs_help = "Number of epochs to train."
    val_split_help = "Validation Split."
    test_split_help = "Test Split. The split must be kept consistent for all experiments."
    split_help = "Makes training and validation split random instead of splitting alphabetically."
    wandb_help = "Toggles weights and biases as logger!"

    parser.add_argument("-e", "--epochs", metavar="N", default=200, type=int, help=epochs_help)
    parser.add_argument("-v", "--val-split", metavar="p", default=0.15, type=float, help=val_split_help)
    parser.add_argument("-t", "--test-split", metavar="p", default=0.15, type=float, help=test_split_help)
    parser.add_argument("-random-split", default=False, action="store_true", help=split_help)
    parser.add_argument("-wandb", default=False, action="store_true", help=wandb_help)

    args = parser.parse_args()

    ModelPaths.archive_latest_vae_model()

    torch.set_float32_matmul_precision("high")

    tno_dataset = TnoDatasetCollection(
        test_split=args.test_split,
        val_split=args.val_split,
        random=args.random_split,
    )

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

    tensorboard_logger = TensorBoardLogger(save_dir=ModelPaths.VAE_LATEST, name="logs", version="tensorboard")
    csv_logger = CSVLogger(save_dir=ModelPaths.VAE_LATEST, name="logs", version="csv")
    loggers = [tensorboard_logger, csv_logger]
    if args.wandb:
        wandb_logger = WandbLogger(project=WANDB_PROJECT_NAME, save_dir=ModelPaths.VAE_LATEST / "logs")
        loggers.append(wandb_logger)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_ssim",
        mode="max",  # model with the smallest validation loss is saved
        dirpath=ModelPaths.VAE_LATEST_CHECKPOINTS,
        filename="{epoch}-{val_loss:.2f}-{val_ssim:.2f}",
    )

    trainer = Trainer(devices=[0], max_epochs=args.epochs, callbacks=[checkpoint_callback], logger=loggers)
    trainer.fit(model=vae, train_dataloaders=train_data, val_dataloaders=val_data)

    logger.info("Training done!")
