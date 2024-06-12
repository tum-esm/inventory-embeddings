import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader

from embeddings.common.log import logger
from embeddings.common.paths import ModelPathsCreator
from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection
from embeddings.models.vae.vae import VariationalAutoEncoder


def finetune() -> None:
    base_model_name = "256"
    city = "Munich"

    base_model_path = ModelPathsCreator.get_vae_model(base_model_name)
    target_model_path = ModelPathsCreator.get_vae_model(f"{base_model_name}_fine_tuned_on_{city.lower()}")
    target_model_path.archive()

    torch.set_float32_matmul_precision("high")

    base_model = VariationalAutoEncoder.load_from_checkpoint(base_model_path.checkpoint)

    dataset_with_city = TnoDatasetCollection().get_case_study_data(city)

    train_data = DataLoader(
        dataset=dataset_with_city,
        batch_size=8,
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
        max_epochs=30,
        callbacks=[checkpoint_callback],
        logger=loggers,
        gradient_clip_val=0.5,
    )
    trainer.fit(model=base_model, train_dataloaders=train_data)

    logger.info("Fine-tuning done!")
