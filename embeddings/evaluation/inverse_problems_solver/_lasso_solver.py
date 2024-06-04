from sklearn.linear_model import Lasso
from torch import Tensor

from embeddings.evaluation.inverse_problem import InverseProblem

from ._inverse_problem_solver import InverseProblemSolver


class LassoSolver(InverseProblemSolver):
    def solve(self, inverse_problem: InverseProblem) -> Tensor:
        lasso = Lasso(alpha=0.1, max_iter=10_000)
        lasso.fit(inverse_problem.A, inverse_problem.y)
        return Tensor(lasso.coef_)
