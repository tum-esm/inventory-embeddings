import numpy as np
from alive_progress import alive_bar

from embeddings.common.csv_writer import CsvWriter
from embeddings.common.paths import ExperimentPaths
from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection
from embeddings.evaluation.compressed_sensing_experiment import CompressedSensingExperiment
from embeddings.evaluation.inverse_problems_solver import GenerativeModelSolver

if __name__ == "__main__":
    dataset = TnoDatasetCollection(deterministic=True).validation_data
    dataset.disable_temporal_transforms()

    measurements = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 12500]

    solver = GenerativeModelSolver()

    ExperimentPaths.archive_latest_evaluation()
    evaluation_csv = CsvWriter(path=ExperimentPaths.LATEST_EVALUATION / "evaluation.csv")

    evaluation_csv.write_header("Measurements", "MSE")

    with alive_bar(len(measurements) * len(dataset)) as bar:
        for num_measurements in measurements:
            for x in dataset:
                experiment = CompressedSensingExperiment(num_measurements=num_measurements)
                x_rec = experiment.solve_random_inverse_problem(x=x, solver=solver)
                mse = float(np.square(np.subtract(x, x_rec)).mean())
                evaluation_csv.write_row(num_measurements, mse)
                bar()
