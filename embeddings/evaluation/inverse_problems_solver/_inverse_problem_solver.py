from abc import ABC, abstractmethod

from torch import Tensor


class InverseProblemSolver(ABC):
    @abstractmethod
    def solve(self, A: Tensor, y: Tensor) -> Tensor: ...  # noqa: N803
