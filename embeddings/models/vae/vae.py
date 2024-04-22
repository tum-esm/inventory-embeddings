import torch
from torch import Tensor, nn

from embeddings.common.gnfr_sector import NUM_GNFR_SECTORS


class _EncoderLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self._layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
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
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._layers(x)


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._layers = nn.Sequential(
            _EncoderLayer(in_channels=NUM_GNFR_SECTORS, out_channels=32),
            _EncoderLayer(in_channels=32, out_channels=64),
            _EncoderLayer(in_channels=64, out_channels=128),
            _EncoderLayer(in_channels=128, out_channels=128),
            nn.Flatten(),
        )
        self._fully_connected_mean = nn.Linear(512, 1024)
        self._fully_connected_var = nn.Linear(512, 1024)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self._layers(x)
        mean = self._fully_connected_mean(h)
        log_var = self._fully_connected_var(h)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._fully_connected_input = nn.Linear(1024, 512)
        self._layers = nn.Sequential(
            _DecoderLayer(in_channels=128, out_channels=128),
            _DecoderLayer(in_channels=128, out_channels=64),
            _DecoderLayer(in_channels=64, out_channels=32),
            _DecoderLayer(in_channels=32, out_channels=NUM_GNFR_SECTORS),
        )

    def forward(self, z: Tensor) -> Tensor:
        intermediate = self._fully_connected_input(z)
        intermediate = intermediate.view(-1, 128, 2, 2)
        return self._layers(intermediate)


class VariationalAutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._encoder = Encoder()
        self._decoder = Decoder()

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
