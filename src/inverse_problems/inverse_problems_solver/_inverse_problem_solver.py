from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor

from src.inverse_problems.inverse_problem import InverseProblem


class InverseProblemSolver(ABC):
    @abstractmethod
    def solve(self, inverse_problem: InverseProblem, **settings: dict[str, Any]) -> Tensor: ...
