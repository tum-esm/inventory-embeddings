import argparse

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader

from src.common.log import logger
from src.common.paths import ModelPathsCreator
from src.dataset.emission_field_transforms import RandomSparseEmittersTransform
from src.dataset.tno_dataset_collection import TnoDatasetCollection
from src.models.vae.vae import VariationalAutoEncoder


def finetune() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    epochs_help = "Number of epochs to train."
    city_help = "Name of the city the model is fine-tuned on."
    base_model_help = "Name of the base model that is fine-tuned, e.g. '1024'."

    parser.add_argument("-e", "--epochs", metavar="N", default=20, type=int, help=epochs_help)
    parser.add_argument("-c", "--city", metavar="C", type=str, help=city_help, required=True)
    parser.add_argument("-b", "--base-model", metavar="M", type=str, help=base_model_help, required=True)

    args = parser.parse_args()
    base_model_name = args.base_model
    city = args.city

    torch.set_float32_matmul_precision("high")

    base_model = VariationalAutoEncoder.load(model_name=base_model_name)

    dataset_with_city = TnoDatasetCollection().get_case_study_data(city, year=2015)
    dataset_with_city.add_sampling_transform(RandomSparseEmittersTransform(lam=100))

    target_model_path = ModelPathsCreator.get_vae_model(f"{base_model_name}_{city.lower()}")
    target_model_path.archive()

    train_data = DataLoader(
        dataset=dataset_with_city,
        batch_size=4,
        shuffle=True,
        num_workers=16,
    )

    tensorboard_logger = TensorBoardLogger(save_dir=target_model_path.base_path, name="logs", version="tensorboard")
    csv_logger = CSVLogger(save_dir=target_model_path.base_path, name="logs", version="csv")
    loggers = [tensorboard_logger, csv_logger]

    checkpoint_callback = ModelCheckpoint(
        monitor="train_ssim",
        mode="max",  # model with the biggest similarity
        dirpath=target_model_path.checkpoints,
        filename="{epoch}-{train_loss:.2f}-{train_ssim:.2f}",
    )

    base_model.learning_rate = 1e-5

    trainer = Trainer(
        devices=[0],
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        logger=loggers,
        gradient_clip_val=0.5,
    )
    trainer.fit(model=base_model, train_dataloaders=train_data)

    logger.info("Fine-tuning done!")
