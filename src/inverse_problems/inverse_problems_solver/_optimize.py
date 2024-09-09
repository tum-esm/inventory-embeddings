import warnings

import cvxpy as cp
import numpy as np

from src.common.log import logger


def optimize(
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
