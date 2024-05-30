import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn

from embeddings.common.gnfr_sector import NUM_GNFR_SECTORS, GnfrSector
from embeddings.models.common.layers import ConvLayer, ConvTransposeLayer, ResidualConvLayer
from embeddings.models.common.metrics import mse, ssim


class Encoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self._layers = nn.Sequential(
            ResidualConvLayer(NUM_GNFR_SECTORS, dropout=0.2),  # 15x32x32
            ResidualConvLayer(NUM_GNFR_SECTORS, dropout=0.2),
            ResidualConvLayer(NUM_GNFR_SECTORS, dropout=0.2),
            ConvLayer(NUM_GNFR_SECTORS, 30, kernel=2, stride=2),  # 30x16x16
            ResidualConvLayer(30, dropout=0.2),
            ResidualConvLayer(30, dropout=0.2),
            ResidualConvLayer(30, dropout=0.2),
            ConvLayer(30, 60, kernel=2, stride=2),  # 60x8x8
            ResidualConvLayer(60, dropout=0.2),
            ResidualConvLayer(60, dropout=0.2),
            ResidualConvLayer(60, dropout=0.2),
        )
        self._fully_connected_mean = nn.Linear(60 * 8 * 8, latent_dim)
        self._fully_connected_var = nn.Linear(60 * 8 * 8, latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self._layers(x)
        h = h.view(-1, 60 * 8 * 8)
        mean = self._fully_connected_mean(h)
        log_var = self._fully_connected_var(h)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self._fully_connected_input = nn.Linear(latent_dim, 60 * 8 * 8)
        self._layers = nn.Sequential(
            ResidualConvLayer(60, dropout=0.2),  # 60x8x8
            ResidualConvLayer(60, dropout=0.2),
            ResidualConvLayer(60, dropout=0.2),
            ConvTransposeLayer(60, 30, kernel=2, stride=2),  # 30x16x16
            ResidualConvLayer(30, dropout=0.2),
            ResidualConvLayer(30, dropout=0.2),
            ResidualConvLayer(30, dropout=0.2),
            ConvTransposeLayer(30, NUM_GNFR_SECTORS, kernel=2, stride=2),  # 15x32x32
            ResidualConvLayer(NUM_GNFR_SECTORS, dropout=0.2),
            ResidualConvLayer(NUM_GNFR_SECTORS, dropout=0.2),
            ResidualConvLayer(NUM_GNFR_SECTORS, batch_norm=False),
        )

    def forward(self, z: Tensor) -> Tensor:
        intermediate = self._fully_connected_input(z)
        intermediate = intermediate.view(-1, 60, 8, 8)
        return self._layers(intermediate)


class VariationalAutoEncoder(LightningModule):
    LATENT_DIMENSION = 256

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
        return torch.optim.Adam(self.parameters(), lr=learning_rate, amsgrad=True)

    def _log_mse_per_sector(self, log_prefix: str, x_batch: Tensor, x_hat_batch: Tensor) -> None:
        for sector in GnfrSector:
            sector_mse = mse(x=x_batch, x_hat=x_hat_batch, channel=sector.to_index())
            self.log(f"{log_prefix}_mse_{sector}", sector_mse / x_batch.size(0), on_step=False, on_epoch=True)

    def training_step(self, x_batch: Tensor) -> Tensor:
        x_hat_batch, mean_batch, log_var_batch = self.forward(x_batch)

        train_loss = self.loss(
            x_hat=x_hat_batch,
            x=x_batch,
            mean=mean_batch,
            log_var=log_var_batch,
        )
        train_ssim = ssim(x=x_batch, x_hat=x_hat_batch)
        train_mse = mse(x=x_batch, x_hat=x_hat_batch)

        batch_size = x_batch.size(0)

        self.log("train_loss", train_loss / batch_size, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_ssim", train_ssim, on_step=False, on_epoch=True)
        self.log("train_mse", train_mse / batch_size, on_step=False, on_epoch=True)
        self._log_mse_per_sector(log_prefix="train", x_batch=x_batch, x_hat_batch=x_hat_batch)

        return train_loss

    def validation_step(self, x_val_batch: Tensor) -> Tensor:
        x_val_hat_batch, mean_batch, log_var_batch = self.forward(x_val_batch)

        val_loss = self.loss(
            x_hat=x_val_hat_batch,
            x=x_val_batch,
            mean=mean_batch,
            log_var=log_var_batch,
        )
        val_ssim = ssim(x=x_val_batch, x_hat=x_val_hat_batch)
        val_mse = mse(x=x_val_batch, x_hat=x_val_hat_batch)

        batch_size = x_val_batch.size(0)

        self.log("val_loss", val_loss / batch_size)
        self.log("val_ssim", val_ssim)
        self.log("val_mse", val_mse / batch_size)
        self._log_mse_per_sector(log_prefix="val", x_batch=x_val_batch, x_hat_batch=x_val_hat_batch)

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
