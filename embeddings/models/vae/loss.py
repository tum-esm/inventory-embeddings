import torch
from torch import Tensor


def loss(x: Tensor, x_hat: Tensor, mean: Tensor, log_var: Tensor) -> Tensor:
    reproduction_loss = 0.1 * torch.nn.MSELoss(reduction="sum")(x_hat, x)
    kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + kld
