import torch
from alive_progress import alive_bar
from torch import Tensor, optim
from torch.utils.data import DataLoader

from embeddings.common.log import logger
from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection
from embeddings.models.device import device as auto_device
from embeddings.models.vae.loss import loss
from embeddings.models.vae.vae import VariationalAutoEncoder


class VaeTrainer:
    def __init__(self, device: str | None = None) -> None:
        tno_dataset = TnoDatasetCollection()

        self._batch_size = 32
        self._val_batch_size = 128
        self._learning_rate = 1e-2

        self._device = device if device else auto_device

        self._train_data = DataLoader(
            dataset=tno_dataset.training_data,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=1,
        )

        self._val_data = DataLoader(
            dataset=tno_dataset.validation_data,
            batch_size=self._val_batch_size,
            shuffle=True,
            num_workers=4,
        )

        self._vae = VariationalAutoEncoder().to(self._device)

        self._vae_loss = loss

        self._optimizer = optim.Adam(self._vae.parameters(), lr=self._learning_rate)

    def train(self, epochs: int) -> VariationalAutoEncoder:
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            logger.info(f"Starting epoch {epoch + 1}...")
            self._vae.train(True)
            train_loss = self._train_epoch()

            self._vae.eval()
            val_loss = self._validation()
            logger.info(f"Average training loss:\t\t{train_loss}")
            logger.info(f"Average validation loss:\t{val_loss}")

            train_losses.append(train_loss)
            val_losses.append(val_loss)

        return self._vae

    def _train_epoch(self) -> float:
        epoch_loss = 0.0
        with alive_bar(len(self._train_data)) as bar:
            for x_batch in self._train_data:
                epoch_loss += self._train_step(x_batch=x_batch.to(self._device))
                bar()
        return epoch_loss / len(self._train_data)

    def _train_step(self, x_batch: Tensor) -> float:
        self._optimizer.zero_grad()

        x_hat_batch, mean_batch, log_var_batch = self._vae(x_batch)

        train_loss = loss(
            x_hat=x_hat_batch,
            x=x_batch,
            mean=mean_batch,
            log_var=log_var_batch,
        )

        train_loss.backward()
        self._optimizer.step()

        return train_loss.item() / self._batch_size

    def _validation(self) -> float:
        val_loss = 0.0
        with alive_bar(total=len(self._val_data)) as bar, torch.no_grad():
            for x_val_batch in self._val_data:
                val_loss += self._val_step(x_val_batch=x_val_batch.to(self._device))
                bar()
        return val_loss / len(self._val_data)

    def _val_step(self, x_val_batch: Tensor) -> float:
        x_val_hat_batch, mean_batch, log_var_batch = self._vae(x_val_batch)

        val_loss = loss(
            x_hat=x_val_hat_batch,
            x=x_val_batch,
            mean=mean_batch,
            log_var=log_var_batch,
        )

        return val_loss.item() / self._val_batch_size
