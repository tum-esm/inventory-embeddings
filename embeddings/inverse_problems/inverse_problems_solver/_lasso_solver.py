from sklearn.linear_model import Lasso
from torch import Tensor

from embeddings.inverse_problems.inverse_problem import InverseProblem

from ._inverse_problem_solver import InverseProblemSolver
from ._sparsity_transforms import DctTransform, DwtTransform, SparsityTransform

_UNKNOWN_TRANSFORM_ERROR = "Set transform is not implemented!"


class LassoSolver(InverseProblemSolver):
    _MAX_ITER = 100_000

    def __init__(self, alpha: float = 0.1, transform: SparsityTransform | None = None) -> None:
        self._alpha = alpha
        self._transform = transform

    def solve(self, inverse_problem: InverseProblem) -> Tensor:
        A = inverse_problem.A.numpy()  # noqa: N806
        y = inverse_problem.y.numpy()

        lasso = Lasso(alpha=self._alpha, max_iter=self._MAX_ITER)
        if self._transform is None:
            lasso.fit(A, y)
            x = lasso.coef_
        elif self._transform is SparsityTransform.DWT:
            A_DWT, coefficient_slices = DwtTransform.transform_sensing_matrix(sensing_matrix=A)  # noqa: N806
            lasso.fit(A_DWT, inverse_problem.y)
            x = DwtTransform.inverse_transform_field(coefficients=lasso.coef_, coefficient_slices=coefficient_slices)
        elif self._transform is SparsityTransform.DCT:
            A_cosine_basis = DctTransform.transform_sensing_matrix(sensing_matrix=A)  # noqa: N806
            lasso.fit(A_cosine_basis, y)
            x = DctTransform.inverse_transform_field(transformed_field_as_vec=lasso.coef_)
        else:
            raise NotImplementedError(_UNKNOWN_TRANSFORM_ERROR)
        return Tensor(x)
