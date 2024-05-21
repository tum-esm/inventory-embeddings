import torch
from torch import Tensor
from torchmetrics.image import StructuralSimilarityIndexMeasure


def ssim(x: Tensor, x_hat: Tensor, channel: int | None = None) -> Tensor:
    metric = StructuralSimilarityIndexMeasure().to(x.device)
    if channel:
        return metric(x[:, channel : channel + 1, :, :], x_hat[:, channel : channel + 1, :, :])
    return metric(x, x_hat)


def mse(x: Tensor, x_hat: Tensor, channel: int | None = None) -> Tensor:
    metric = torch.nn.MSELoss(reduction="sum")
    if channel is not None:
        return metric(x[:, channel : channel + 1, :, :], x_hat[:, channel : channel + 1, :, :])
    return metric(x, x_hat)
