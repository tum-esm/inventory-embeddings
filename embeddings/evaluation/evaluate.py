import numpy as np
from tqdm import tqdm

from embeddings.common.csv_writer import CsvWriter
from embeddings.common.paths import ExperimentPaths
from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection
from embeddings.evaluation.compressed_sensing_experiment import generate_random_inverse_problem, solve_inverse_problem
from embeddings.evaluation.inverse_problems_solver import DwtLassoSolver, GenerativeModelSolver, LassoSolver
from embeddings.models.common.metrics import ssim

NUM_MEASUREMENTS = [50, 100, 250, 500, 1000, 2500, 5000, 10000, 12500]
NUM_EXPERIMENTS = 3


def evaluate() -> None:
    dataset = TnoDatasetCollection().test_data
    dataset.disable_temporal_transforms()

    solvers = {
        "Generative": GenerativeModelSolver(),
        "Lasso": LassoSolver(),
        "Lasso_DWT": DwtLassoSolver(),
    }

    ExperimentPaths.archive_latest_evaluation()

    for run_index in tqdm(range(NUM_EXPERIMENTS), desc="Run"):
        evaluation_csv = CsvWriter(path=ExperimentPaths.EVALUATION_LATEST / f"evaluation_{run_index}.csv")
        evaluation_csv.write_header("Measurements", "Solver", "MSE", "SSIM")

        with tqdm(total=len(NUM_MEASUREMENTS) * len(dataset), leave=False, desc="Compressed Sensing") as bar:
            for num_measurements in NUM_MEASUREMENTS:
                for x in dataset:
                    inverse_problem = generate_random_inverse_problem(x=x, num_measurements=num_measurements)
                    for name, solver in solvers.items():
                        x_rec = solve_inverse_problem(solver=solver, inverse_problem=inverse_problem)
                        mse = float(np.square(np.subtract(x, x_rec)).mean())
                        s = float(ssim(x=x, x_hat=x_rec))
                        evaluation_csv.write_row(num_measurements, name, mse, s)
                    bar.update()
