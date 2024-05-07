import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn


class Encoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self._layers = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self._fully_connected_mean = nn.Linear(128, latent_dim)
        self._fully_connected_var = nn.Linear(128, latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self._layers(x)
        mean = self._fully_connected_mean(h)
        log_var = self._fully_connected_var(h)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self._layers = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 1024),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self._layers(z)


class VariationalAutoEncoder(LightningModule):
    LATENT_DIMENSION = 100

    def __init__(self) -> None:
        super().__init__()
        self._encoder = Encoder(latent_dim=self.LATENT_DIMENSION)
        self._decoder = Decoder(latent_dim=self.LATENT_DIMENSION)

    @staticmethod
    def loss(x: Tensor, x_hat: Tensor, mean: Tensor, log_var: Tensor) -> Tensor:
        reproduction_loss = torch.nn.MSELoss(reduction="sum")(x_hat, x)
        kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return reproduction_loss + kld

    @property
    def encoder(self) -> Encoder:
        return self._encoder

    @property
    def decoder(self) -> Decoder:
        return self._decoder

    def _reparameterization(self, mean: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mean

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x_vector = x.view(-1, 1024)
        mean, log_var = self._encoder(x_vector)
        z = self._reparameterization(mean, log_var)
        x_hat_vector = self._decoder(z)
        x_hat = x_hat_vector.view(-1, 32, 32)
        return x_hat, mean, log_var

    def configure_optimizers(self) -> OptimizerLRScheduler:
        learning_rate = 1e-5
        return torch.optim.Adam(self.parameters(recurse=True), lr=learning_rate)

    def training_step(self, x_batch: Tensor) -> Tensor:
        x_hat_batch, mean_batch, log_var_batch = self.forward(x_batch)

        train_loss = self.loss(
            x_hat=x_hat_batch,
            x=x_batch,
            mean=mean_batch,
            log_var=log_var_batch,
        )

        batch_size = x_batch.size(0)

        self.log("train_loss", train_loss / batch_size, on_step=False, on_epoch=True, prog_bar=True)

        return train_loss

    def validation_step(self, x_val_batch: Tensor) -> Tensor:
        x_val_hat_batch, mean_batch, log_var_batch = self.forward(x_val_batch)

        val_loss = self.loss(
            x_hat=x_val_hat_batch,
            x=x_val_batch,
            mean=mean_batch,
            log_var=log_var_batch,
        )

        batch_size = x_val_batch.size(0)

        self.log("validation_loss", val_loss / batch_size)

        return val_loss

    def reconstruct(self, x: Tensor) -> Tensor:
        self.eval()
        in_tensor = x.unsqueeze(0).to(self.device)
        with torch.no_grad():
            out_tensor, _, _ = self.forward(in_tensor)
        return out_tensor.squeeze(0).cpu()

    def generate(self) -> Tensor:
        self.eval()
        with torch.no_grad():
            noise = torch.randn(1, self.LATENT_DIMENSION).to(self.device)
            generated = self.decoder(noise)
        return generated.squeeze(0).cpu()
