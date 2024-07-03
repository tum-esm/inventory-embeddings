import numpy as np
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

def relative_error(x: Tensor, x_hat: Tensor, channel: int | None = None) -> Tensor:
    three_dimensions = 3

    if x.dim() != three_dimensions:
        x = x.squeeze(0)
    if x_hat.dim() != three_dimensions:
        x_hat = x_hat.squeeze(0)

    x_np = np.array(x)
    x_hat_np = np.array(x_hat)

    if channel is not None:
        x_np = x_np[:, channel : channel + 1, :, :]
        x_hat_np = x_hat_np[:, channel : channel + 1, :, :]

    p = 2

    error_norm = np.linalg.norm((x_np - x_hat_np).flatten(), ord=p)

    x_norm = np.linalg.norm(x_np.flatten(), ord=p)

    relative_error = error_norm / x_norm

    return torch.tensor(relative_error)
