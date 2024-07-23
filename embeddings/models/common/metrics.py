import numpy as np
import torch
from torch import Tensor
from torchmetrics.image import StructuralSimilarityIndexMeasure

_TWO = 2
_THREE = 3

_CHANNEL_ERROR_MESSAGE = "Cannot provide channel if number of dimensions is only 2!"

_SSIM = StructuralSimilarityIndexMeasure()

def ssim(x: Tensor, x_hat: Tensor, channel: int | None = None) -> Tensor:
    if x.ndimension() == _TWO:
        if channel is not None:
            raise AttributeError(_CHANNEL_ERROR_MESSAGE)
        x = x.unsqueeze(0).unsqueeze(0)
        x_hat = x_hat.unsqueeze(0).unsqueeze(0)
    elif x.ndimension() == _THREE:
        x = x.unsqueeze(0)
        x_hat = x_hat.unsqueeze(0)
    if channel:
        x = x[:, channel : channel + 1, :, :]
        x_hat = x_hat[:, channel : channel + 1, :, :]

    if _SSIM.device != x.device:
        _SSIM.to(x.device)
    return _SSIM(x, x_hat)


def mse(x: Tensor, x_hat: Tensor, channel: int | None = None) -> Tensor:
    if x.ndimension() == _TWO:
        if channel is not None:
            raise AttributeError(_CHANNEL_ERROR_MESSAGE)
        x = x.unsqueeze(0).unsqueeze(0)
        x_hat = x_hat.unsqueeze(0).unsqueeze(0)
    elif x.ndimension() == _THREE:
        x = x.unsqueeze(0)
        x_hat = x_hat.unsqueeze(0)
    if channel:
        x = x[:, channel : channel + 1, :, :]
        x_hat = x_hat[:, channel : channel + 1, :, :]

    metric = torch.nn.MSELoss(reduction="mean")
    if channel is not None:
        return metric(x[:, channel : channel + 1, :, :], x_hat[:, channel : channel + 1, :, :])
    return metric(x, x_hat)

def relative_error(x: Tensor, x_hat: Tensor, channel: int | None = None) -> Tensor:
    if x.ndimension() == _TWO:
        if channel is not None:
            raise AttributeError(_CHANNEL_ERROR_MESSAGE)
    elif x.ndimension() == _THREE:
        x = x.unsqueeze(0)
        x_hat = x_hat.unsqueeze(0)
    if channel:
        x = x[:, channel : channel + 1, :, :]
        x_hat = x_hat[:, channel : channel + 1, :, :]

    x_np = np.array(x)
    x_hat_np = np.array(x_hat)

    p = 2

    error_norm = np.linalg.norm((x_np - x_hat_np).flatten(), ord=p)

    x_norm = np.linalg.norm(x_np.flatten(), ord=p)

    relative_error = error_norm / x_norm

    return torch.tensor(relative_error)
