import torch
from torch import Tensor
from torchmetrics.image import StructuralSimilarityIndexMeasure

_SSIM = StructuralSimilarityIndexMeasure()


def ssim(x: Tensor, x_hat: Tensor, channel: int | None = None) -> Tensor:
    if _SSIM.device != x.device:
        _SSIM.to(x.device)
    if channel:
        return _SSIM(x[:, channel : channel + 1, :, :], x_hat[:, channel : channel + 1, :, :])
    return _SSIM(x, x_hat)


def mse(x: Tensor, x_hat: Tensor, channel: int | None = None) -> Tensor:
    metric = torch.nn.MSELoss(reduction="sum")
    if channel is not None:
        return metric(x[:, channel : channel + 1, :, :], x_hat[:, channel : channel + 1, :, :])
    return metric(x, x_hat)
