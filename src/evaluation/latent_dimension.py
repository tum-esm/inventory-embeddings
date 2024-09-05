from src.common.paths import ExperimentPaths
from src.dataset.tno_dataset_collection import TnoDatasetCollection
from src.evaluation.evaluation import Evaluation, EvaluationSettings
from src.inverse_problems.inverse_problems_solver import GenerativeModelSolver, InverseProblemSolver

if __name__ == "__main__":
    test_data = TnoDatasetCollection().test_data

    solvers: dict[str, InverseProblemSolver] = {
        "VAE 256": GenerativeModelSolver.from_vae_model_name(name="256"),
        "VAE 512": GenerativeModelSolver.from_vae_model_name(name="512"),
        "VAE 1024": GenerativeModelSolver.from_vae_model_name(name="1024"),
        "VAE 2048": GenerativeModelSolver.from_vae_model_name(name="2048"),
    }

    settings = EvaluationSettings(
        base_path=ExperimentPaths.LATENT_DIMENSION,
        measurements=[
            100,
            250,
            500,
            1_000,
            2_500,
            5_000,
            10_000,
            12_500,
        ],
        snr=[100],
        dataset=test_data,
    )

    evaluation = Evaluation(settings)

    evaluation.run(solvers)
