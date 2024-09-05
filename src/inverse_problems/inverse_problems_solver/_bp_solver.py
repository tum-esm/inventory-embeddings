import warnings

import cvxpy as cp
import numpy as np
from torch import Tensor

from src.common.log import logger
from src.inverse_problems.inverse_problem import InverseProblem

from ._inverse_problem_solver import InverseProblemSolver
from ._sparsity_transforms import DctTransform, DwtTransform, SparsityTransform

_UNKNOWN_TRANSFORM_ERROR = "Set transform is not implemented!"


def _optimize(
    A: np.ndarray,  # noqa: N803
    b: np.ndarray,
    error: np.ndarray | None,
    p: int = 1,
    *,
    verbose: bool = True,
) -> np.ndarray:
    n = A.shape[1]

    x_res = cp.Variable(n)

    objective = cp.Minimize(cp.norm(x_res, p))

    error_sum_squares = 0 if error is None else np.sum(error**2)

    constraints = [cp.sum_squares(A @ x_res - b) <= error_sum_squares]

    options_gurobi = {
        "Method": 2,  # Barrier method, which is often suitable for convex problems.
        "BarHomogeneous": 1,
        "BarConvTol": 1e-8,  # Convergence tolerance for barrier method.
        "BarQCPConvTol": 1e-8,
        "TimeLimit": 600,  # Limit the time for solving (in seconds).
        "OutputFlag": 1 if verbose else 0,  # 0 suppresses output.
    }

    prob = cp.Problem(objective, constraints)
    with warnings.catch_warnings(action="ignore"):
        try:
            prob.solve(solver=cp.GUROBI, verbose=verbose, **options_gurobi)
        except (cp.error.SolverError, ImportError) as e:
            logger.warning(f"Gurobi failed with error: {e}")
            logger.warning("Using default solver instead!")
            prob.solve(verbose=verbose)
    return x_res.value


class BasisPursuitSolver(InverseProblemSolver):
    """
    In case of noisy measurements, basis pursuit denoising is used!
    """

    def __init__(self, transform: SparsityTransform | None = None, verbose: bool = False) -> None:
        self._transform = transform
        self._verbose = verbose

    def solve(self, inverse_problem: InverseProblem) -> Tensor:
        A = inverse_problem.A.numpy()  # noqa: N806
        y = inverse_problem.y.numpy()
        noise = inverse_problem.noise.numpy() if inverse_problem.noise is not None else None

        if self._transform is None:
            x = _optimize(A, y, noise, p=1, verbose=self._verbose)
        elif self._transform is SparsityTransform.DWT:
            A_DWT, coefficient_slices = DwtTransform.transform_sensing_matrix(sensing_matrix=A)  # noqa: N806
            res = _optimize(A_DWT, y, noise, p=1, verbose=self._verbose)
            x = DwtTransform.inverse_transform_field(coefficients=res, coefficient_slices=coefficient_slices)
        elif self._transform is SparsityTransform.DCT:
            A_cosine_basis = DctTransform.transform_sensing_matrix(sensing_matrix=A)  # noqa: N806
            res = _optimize(A_cosine_basis, y, noise, p=1, verbose=self._verbose)
            x = DctTransform.inverse_transform_field(transformed_field_as_vec=res)
        else:
            raise NotImplementedError(_UNKNOWN_TRANSFORM_ERROR)
        return Tensor(x)
