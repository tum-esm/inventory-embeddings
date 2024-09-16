from typing import Any

from torch import Tensor

from src.inverse_problems.inverse_problem import InverseProblem

from ._inverse_problem_solver import InverseProblemSolver
from ._optimize import optimize

_UNKNOWN_TRANSFORM_ERROR = "Set transform is not implemented!"


class LeastSquaresSolver(InverseProblemSolver):
    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose

    def solve(self, inverse_problem: InverseProblem, **_settings: dict[str, Any]) -> Tensor:
        A = inverse_problem.A.numpy()  # noqa: N806
        y = inverse_problem.y.numpy()
        noise = inverse_problem.noise.numpy() if inverse_problem.noise is not None else None

        x = optimize(A, y, noise, p=2, verbose=self._verbose)

        return Tensor(x)
