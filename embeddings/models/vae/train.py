import torch

from embeddings.common.log import logger
from embeddings.common.paths import ModelPaths
from embeddings.models.vae.vae_trainer import VaeTrainer


def train() -> None:
    ModelPaths.archive_latest_vae_model()

    vae_trainer = VaeTrainer()
    vae = vae_trainer.train(epochs=30)

    logger.info(f"Training done! Saving model to {ModelPaths.VAE_LATEST_MODEL}")
    torch.save(vae.state_dict(), ModelPaths.VAE_LATEST_MODEL)
