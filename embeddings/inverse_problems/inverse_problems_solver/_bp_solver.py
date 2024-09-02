import cvxpy as cp
import torch
from torch import Tensor

from embeddings.inverse_problems.inverse_problem import InverseProblem

from ._inverse_problem_solver import InverseProblemSolver


def _optimize(A: Tensor, b: Tensor, error: Tensor | None, p: int = 1) -> Tensor:  # noqa: N803
    n = A.shape[1]

    x_res = cp.Variable(n)

    objective = cp.Minimize(cp.norm(x_res, p))

    constraints = [A @ x_res == b] if error is None else [cp.sum_squares(A @ x_res - b) <= torch.sum(error**2).item()]

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=True)

    return x_res.value


class BasisPursuitSolver(InverseProblemSolver):
    """
    In case of noisy measurements, basis pursuit denoising is used!
    """

    def solve(self, inverse_problem: InverseProblem) -> Tensor:
        res = _optimize(inverse_problem.A, inverse_problem.y, inverse_problem.noise, p=1)
        return Tensor(res)
