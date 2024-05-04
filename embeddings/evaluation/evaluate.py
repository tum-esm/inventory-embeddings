import numpy as np
from alive_progress import alive_bar
from matplotlib import pyplot as plt

from embeddings.common.paths import PlotPaths
from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection
from embeddings.evaluation.compressed_sensing_experiment import CompressedSensingExperiment
from embeddings.evaluation.inverse_problems_solver import GenerativeModelSolver

if __name__ == "__main__":
    dataset = TnoDatasetCollection(deterministic=True).validation_data
    dataset.disable_temporal_transforms()

    measurements = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 12500]
    reconstruction_errors = []

    solver = GenerativeModelSolver()

    with alive_bar(5 * len(measurements) * len(dataset)) as bar:
        for num_measurements in measurements:
            mse_losses = []

            for x in dataset:
                for _ in range(5):
                    experiment = CompressedSensingExperiment(num_measurements=num_measurements)
                    x_rec = experiment.solve_random_inverse_problem(x=x, solver=solver)
                    mse = np.square(np.subtract(x, x_rec)).mean()
                    mse_losses.append(mse)
                    bar()

            reconstruction_errors.append(np.mean(np.array(mse_losses)))

    plt.plot(measurements, reconstruction_errors)
    plt.xscale("log")
    plt.savefig(PlotPaths.PLOTS / "measurements.png")
