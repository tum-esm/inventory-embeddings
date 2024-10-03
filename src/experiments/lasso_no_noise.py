from src.common.paths import ExperimentPath
from src.dataset.tno_dataset_collection import TnoDatasetCollection
from src.experiments.evaluation.evaluation_runner import EvaluationRunner, EvaluationSettings
from src.inverse_problems.inverse_problems_solver import InverseProblemSolver, LassoSolver, SparsityTransform

EXPERIMENT_NAME = "lasso_no_noise"

SNR = None  # No noise
NUM_MEASUREMENTS = [500, 1000, 2500, 5000, 10000, 12500]

dataset_collection = TnoDatasetCollection()

dataset = dataset_collection.test_data

dataset.disable_temporal_transforms()

solvers: dict[str, InverseProblemSolver] = {
    "lasso": LassoSolver(),
    "lasso_dwt": LassoSolver(transform=SparsityTransform.DWT),
    "lasso_dct": LassoSolver(transform=SparsityTransform.DCT),
}

settings = EvaluationSettings(
    measurements=NUM_MEASUREMENTS,
    snr=[SNR],
    dataset=dataset,
    path=ExperimentPath(EXPERIMENT_NAME),
)

evaluation_runner = EvaluationRunner(settings=settings)

evaluation_runner.run(solvers=solvers)
