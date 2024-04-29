import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn

from embeddings.common.gnfr_sector import NUM_GNFR_SECTORS


class _EncoderLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self._layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._layers(x)


class _DecoderLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self._layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._layers(x)


class Encoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self._layers = nn.Sequential(
            _EncoderLayer(in_channels=NUM_GNFR_SECTORS, out_channels=64),
            _EncoderLayer(in_channels=64, out_channels=128),
            _EncoderLayer(in_channels=128, out_channels=256),
            _EncoderLayer(in_channels=256, out_channels=512),
            nn.Flatten(),
        )
        self._fully_connected_mean = nn.Linear(2048, latent_dim)
        self._fully_connected_var = nn.Linear(2048, latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self._layers(x)
        mean = self._fully_connected_mean(h)
        log_var = self._fully_connected_var(h)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self._fully_connected_input = nn.Linear(latent_dim, 2048)
        self._layers = nn.Sequential(
            _DecoderLayer(in_channels=512, out_channels=256),
            _DecoderLayer(in_channels=256, out_channels=128),
            _DecoderLayer(in_channels=128, out_channels=64),
            _DecoderLayer(in_channels=64, out_channels=NUM_GNFR_SECTORS),
        )

    def forward(self, z: Tensor) -> Tensor:
        intermediate = self._fully_connected_input(z)
        intermediate = intermediate.view(-1, 512, 2, 2)
        return self._layers(intermediate)


class VariationalAutoEncoder(LightningModule):
    LATENT_DIMENSION = 512

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
        mean, log_var = self._encoder(x)
        z = self._reparameterization(mean, log_var)
        x_hat = self._decoder(z)
        return x_hat, mean, log_var

    def configure_optimizers(self) -> OptimizerLRScheduler:
        learning_rate = 1e-3
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
            noise = torch.randn(1, 512).to(self.device)
            generated = self.decoder(noise)
        return generated.squeeze(0).cpu()
