from dataclasses import dataclass

from torch import Tensor


@dataclass
class InverseProblem:
    A: Tensor
    y: Tensor
    noise: Tensor | None
