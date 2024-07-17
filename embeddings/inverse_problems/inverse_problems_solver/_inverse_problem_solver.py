from abc import ABC, abstractmethod

from torch import Tensor

from embeddings.inverse_problems.inverse_problem import InverseProblem


class InverseProblemSolver(ABC):
    @abstractmethod
    def solve(self, inverse_problem: InverseProblem) -> Tensor: ...
