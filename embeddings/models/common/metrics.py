import torch
from torch import Tensor
from torchmetrics.image import StructuralSimilarityIndexMeasure

_SSIM = StructuralSimilarityIndexMeasure()


def ssim(x: Tensor, x_hat: Tensor, channel: int | None = None) -> Tensor:
    three_dimensions = 3

    if _SSIM.device != x.device:
        _SSIM.to(x.device)
    if x.dim() == three_dimensions:
        x = x.unsqueeze(0)
    if x_hat.dim() == three_dimensions:
        x_hat = x_hat.unsqueeze(0)
    if channel:
        return _SSIM(x[:, channel : channel + 1, :, :], x_hat[:, channel : channel + 1, :, :])
    return _SSIM(x, x_hat)


def mse(x: Tensor, x_hat: Tensor, channel: int | None = None) -> Tensor:
    metric = torch.nn.MSELoss(reduction="mean")
    if channel is not None:
        return metric(x[:, channel : channel + 1, :, :], x_hat[:, channel : channel + 1, :, :])
    return metric(x, x_hat)
