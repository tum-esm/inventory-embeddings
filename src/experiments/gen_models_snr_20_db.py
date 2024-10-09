from src.common.paths import ExperimentPath
from src.dataset.tno_dataset_collection import TnoDatasetCollection
from src.experiments.evaluation.evaluation_runner import EvaluationRunner, EvaluationSettings
from src.inverse_problems.inverse_problems_solver import GenerativeModelSolver, InverseProblemSolver

EXPERIMENT_NAME = "gen_models_snr_20_db"

MODEL_256 = "256"
MODEL_512 = "512"
MODEL_1024 = "1024"
MODEL_2048 = "2048"

SNR = 100  # 20 dB
NUM_MEASUREMENTS = [500, 1000, 2500, 5000, 10000, 12500]

dataset_collection = TnoDatasetCollection()

dataset = dataset_collection.test_data

dataset.disable_temporal_transforms()

solvers: dict[str, InverseProblemSolver] = {
    MODEL_256: GenerativeModelSolver.from_vae_model_name(MODEL_256),
    MODEL_512: GenerativeModelSolver.from_vae_model_name(MODEL_512),
    MODEL_1024: GenerativeModelSolver.from_vae_model_name(MODEL_1024),
    MODEL_2048: GenerativeModelSolver.from_vae_model_name(MODEL_2048),
}

settings = EvaluationSettings(
    measurements=NUM_MEASUREMENTS,
    snr=[SNR],
    dataset=dataset,
    path=ExperimentPath(EXPERIMENT_NAME),
    repetitions=3,
)

evaluation_runner = EvaluationRunner(settings=settings)

evaluation_runner.run(solvers=solvers)
