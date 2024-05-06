import numpy as np
from tqdm import tqdm

from embeddings.common.csv_writer import CsvWriter
from embeddings.common.paths import ExperimentPaths
from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection
from embeddings.evaluation.compressed_sensing_experiment import CompressedSensingExperiment
from embeddings.evaluation.inverse_problems_solver import GenerativeModelSolver

NUM_MEASUREMENTS = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 12500]
NUM_EXPERIMENTS = 5

if __name__ == "__main__":
    dataset = TnoDatasetCollection(deterministic=True).validation_data
    dataset.disable_temporal_transforms()

    solver = GenerativeModelSolver()

    ExperimentPaths.archive_latest_evaluation()

    for run_index in tqdm(range(NUM_EXPERIMENTS), desc="Run"):
        run = ExperimentPaths.LATEST_EVALUATION / f"run_{run_index}"
        run.mkdir()
        evaluation_csv = CsvWriter(path=run / "evaluation.csv")
        evaluation_csv.write_header("Measurements", "MSE")

        with tqdm(total=len(NUM_MEASUREMENTS) * len(dataset), leave=False, desc="Compressed Sensing") as bar:
            for num_measurements in NUM_MEASUREMENTS:
                for x in dataset:
                    experiment = CompressedSensingExperiment(num_measurements=num_measurements)
                    x_rec = experiment.solve_random_inverse_problem(x=x, solver=solver)
                    mse = float(np.square(np.subtract(x, x_rec)).mean())
                    evaluation_csv.write_row(num_measurements, mse)
                    bar.update()
