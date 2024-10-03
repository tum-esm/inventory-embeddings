from src.common.paths import ExperimentPath
from src.dataset.tno_dataset_collection import TnoDatasetCollection
from src.experiments.evaluation.evaluation_runner import EvaluationRunner, EvaluationSettings
from src.inverse_problems.inverse_problems_solver import GenerativeModelSolver, InverseProblemSolver

MODEL_256 = "256"
MODEL_512 = "512"
MODEL_1024 = "1024"
MODEL_2048 = "2048"

SNR = None
NUM_MEASUREMENTS = [500, 1000, 2500, 5000, 10000, 12500]

CITIES = ["Munich", "Zürich", "Paris"]

dataset_collection = TnoDatasetCollection()

base_models: dict[str, InverseProblemSolver] = {
    MODEL_256: GenerativeModelSolver.from_vae_model_name(MODEL_256),
    MODEL_512: GenerativeModelSolver.from_vae_model_name(MODEL_512),
    MODEL_1024: GenerativeModelSolver.from_vae_model_name(MODEL_1024),
    MODEL_2048: GenerativeModelSolver.from_vae_model_name(MODEL_2048),
}


def get_fine_tuned_models(c: str) -> dict[str, InverseProblemSolver]:
    return {
        f"{MODEL_256}_{c}": GenerativeModelSolver.from_vae_model_name(f"{MODEL_256}_{c}"),
        f"{MODEL_512}_{c}": GenerativeModelSolver.from_vae_model_name(f"{MODEL_512}_{c}"),
        f"{MODEL_1024}_{c}": GenerativeModelSolver.from_vae_model_name(f"{MODEL_1024}_{c}"),
        f"{MODEL_2048}_{c}": GenerativeModelSolver.from_vae_model_name(f"{MODEL_2048}_{c}"),
    }


for city in CITIES:
    city_lower_case = city.lower()

    dataset = dataset_collection.get_case_study_data(city=city, year=2018)
    dataset.disable_temporal_transforms()

    solvers: dict[str, InverseProblemSolver] = {
        **base_models,
        **get_fine_tuned_models(city_lower_case),
    }

    settings = EvaluationSettings(
        measurements=NUM_MEASUREMENTS,
        snr=[SNR],
        dataset=dataset,
        path=ExperimentPath(name=f"fine_tuned_{city_lower_case}_no_noise"),
    )

    evaluation_runner = EvaluationRunner(settings=settings)

    evaluation_runner.run(solvers=solvers)
