from sklearn.linear_model import Lasso
from torch import Tensor

from embeddings.evaluation.inverse_problem import InverseProblem

from . import InverseProblemSolver


class LassoSolver(InverseProblemSolver):
    def solve(self, inverse_problem: InverseProblem) -> Tensor:
        lasso = Lasso()
        lasso.fit(inverse_problem.A, inverse_problem.y)
        return Tensor(lasso.coef_)
