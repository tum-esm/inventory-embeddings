from torch import Tensor, nn


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        stride: int = 1,
        padding: int = 0,
        activation: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self._layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
        )
        if activation:
            self._layers.append(nn.LeakyReLU())

    def forward(self, x: Tensor) -> Tensor:
        return self._layers(x)


class ConvTransposeLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self._layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._layers(x)


class ResidualConvLayer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self._conv_layers = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return x + self._conv_layers(x)
