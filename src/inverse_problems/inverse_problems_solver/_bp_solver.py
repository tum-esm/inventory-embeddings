from torch import Tensor

from src.inverse_problems.inverse_problem import InverseProblem

from ._inverse_problem_solver import InverseProblemSolver
from ._optimize import optimize
from ._sparsity_transforms import DctTransform, DwtTransform, SparsityTransform

_UNKNOWN_TRANSFORM_ERROR = "Set transform is not implemented!"


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
            x = optimize(A, y, noise, p=1, verbose=self._verbose)
        elif self._transform is SparsityTransform.DWT:
            A_DWT, coefficient_slices = DwtTransform.transform_sensing_matrix(sensing_matrix=A)  # noqa: N806
            res = optimize(A_DWT, y, noise, p=1, verbose=self._verbose)
            x = DwtTransform.inverse_transform_field(coefficients=res, coefficient_slices=coefficient_slices)
        elif self._transform is SparsityTransform.DCT:
            A_cosine_basis = DctTransform.transform_sensing_matrix(sensing_matrix=A)  # noqa: N806
            res = optimize(A_cosine_basis, y, noise, p=1, verbose=self._verbose)
            x = DctTransform.inverse_transform_field(transformed_field_as_vec=res)
        else:
            raise NotImplementedError(_UNKNOWN_TRANSFORM_ERROR)
        return Tensor(x)
