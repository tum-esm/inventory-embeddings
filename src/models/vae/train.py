import argparse

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader

from src.common.log import logger
from src.common.paths import ModelPathsCreator
from src.dataset.tno_dataset_collection import TnoDatasetCollection, TnoDatasetCollectionSettings
from src.models.vae.vae import VariationalAutoEncoder

WANDB_PROJECT_NAME = "inventory-embeddings-vae"


def train() -> None:
    settings = TnoDatasetCollectionSettings()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    epochs_help = "Number of epochs to train."
    latent_dim_help = "Latent Dimension."
    v_split_help = "Validation Split."
    t_split_help = "Test Split. The split must be kept consistent for all experiments."
    wandb_help = "Toggles weights and biases as logger!"

    parser.add_argument("-e", "--epochs", metavar="N", default=100, type=int, help=epochs_help)
    parser.add_argument("-d", "--latent-dim", metavar="N", default=256, type=int, help=latent_dim_help)
    parser.add_argument("-v", "--val-split", metavar="p", default=settings.val_split, type=float, help=v_split_help)
    parser.add_argument("-t", "--test-split", metavar="p", default=settings.test_split, type=float, help=t_split_help)
    parser.add_argument("-wandb", default=False, action="store_true", help=wandb_help)

    args = parser.parse_args()

    latest_vae_paths = ModelPathsCreator.get_latest_vae_model()
    latest_vae_paths.archive()

    torch.set_float32_matmul_precision("high")

    settings.test_split = args.test_split
    settings.val_split = args.val_split

    tno_dataset = TnoDatasetCollection(settings)

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

    vae = VariationalAutoEncoder(latent_dimension=args.latent_dim)

    tensorboard_logger = TensorBoardLogger(save_dir=latest_vae_paths.base_path, name="logs", version="tensorboard")
    csv_logger = CSVLogger(save_dir=latest_vae_paths.base_path, name="logs", version="csv")
    loggers = [tensorboard_logger, csv_logger]
    if args.wandb:
        wandb_logger = WandbLogger(project=WANDB_PROJECT_NAME, save_dir=latest_vae_paths.logs)
        loggers.append(wandb_logger)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_ssim",
        mode="max",  # model with biggest ssim is stored
        dirpath=latest_vae_paths.checkpoints,
        filename="{epoch}-{val_loss:.2f}-{val_ssim:.2f}",
    )

    trainer = Trainer(
        devices=[0],
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        logger=loggers,
        gradient_clip_val=0.5,
    )
    trainer.fit(model=vae, train_dataloaders=train_data, val_dataloaders=val_data)

    logger.info("Training done!")
