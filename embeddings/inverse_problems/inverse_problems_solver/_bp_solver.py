import cvxpy as cp
import numpy as np
from torch import Tensor

from embeddings.inverse_problems.inverse_problem import InverseProblem

from ._inverse_problem_solver import InverseProblemSolver
from ._sparsity_transforms import DctTransform, DwtTransform, SparsityTransform

_UNKNOWN_TRANSFORM_ERROR = "Set transform is not implemented!"


def _optimize(A: np.ndarray, b: np.ndarray, error: np.ndarray | None, p: int = 1) -> np.ndarray:  # noqa: N803
    n = A.shape[1]

    x_res = cp.Variable(n)

    objective = cp.Minimize(cp.norm(x_res, p))

    constraints = [A @ x_res == b] if error is None else [cp.sum_squares(A @ x_res - b) <= np.sum(error**2)]

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=True)

    return x_res.value


class BasisPursuitSolver(InverseProblemSolver):
    """
    In case of noisy measurements, basis pursuit denoising is used!
    """

    def __init__(self, transform: SparsityTransform | None = None) -> None:
        self._transform = transform

    def solve(self, inverse_problem: InverseProblem) -> Tensor:
        A = inverse_problem.A.numpy()  # noqa: N806
        y = inverse_problem.y.numpy()
        noise = inverse_problem.noise.numpy() if inverse_problem.noise is not None else None

        if self._transform is None:
            x = _optimize(A, y, noise, p=1)
        elif self._transform is SparsityTransform.DWT:
            A_DWT, coefficient_slices = DwtTransform.transform_sensing_matrix(sensing_matrix=A)  # noqa: N806
            res = _optimize(A_DWT, y, noise, p=1)
            x = DwtTransform.inverse_transform_field(coefficients=res, coefficient_slices=coefficient_slices)
        elif self._transform is SparsityTransform.DCT:
            A_cosine_basis = DctTransform.transform_sensing_matrix(sensing_matrix=A)  # noqa: N806
            res = _optimize(A_cosine_basis, y, noise, p=1)
            x = DctTransform.inverse_transform_field(transformed_field_as_vec=res)
        else:
            raise NotImplementedError(_UNKNOWN_TRANSFORM_ERROR)
        return Tensor(x)
