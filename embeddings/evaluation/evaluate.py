import numpy as np
from tqdm import tqdm

from embeddings.common.csv_writer import CsvWriter
from embeddings.common.paths import ExperimentPaths
from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection
from embeddings.evaluation.compressed_sensing_experiment import generate_random_inverse_problem, solve_inverse_problem
from embeddings.evaluation.inverse_problems_solver import GenerativeModelSolver

NUM_MEASUREMENTS = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 12500]
NUM_EXPERIMENTS = 5


def evaluate() -> None:
    dataset = TnoDatasetCollection().test_data
    dataset.disable_temporal_transforms()

    generative_model_solver = GenerativeModelSolver()

    ExperimentPaths.archive_latest_evaluation()

    for run_index in tqdm(range(NUM_EXPERIMENTS), desc="Run"):
        evaluation_csv = CsvWriter(path=ExperimentPaths.LATEST_EVALUATION / f"evaluation_{run_index}.csv")
        evaluation_csv.write_header("Measurements", "MSE")

        with tqdm(total=len(NUM_MEASUREMENTS) * len(dataset), leave=False, desc="Compressed Sensing") as bar:
            for num_measurements in NUM_MEASUREMENTS:
                for x in dataset:
                    inverse_problem = generate_random_inverse_problem(x=x, num_measurements=num_measurements)
                    x_rec = solve_inverse_problem(solver=generative_model_solver, inverse_problem=inverse_problem)
                    mse = float(np.square(np.subtract(x, x_rec)).mean())
                    evaluation_csv.write_row(num_measurements, mse)
                    bar.update()
