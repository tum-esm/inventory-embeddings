import random

import torch
from matplotlib import pyplot as plt

from embeddings.common.paths import PlotPaths
from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection
from embeddings.evaluation.compressed_sensing_experiment import CompressedSensingExperiment
from embeddings.evaluation.inverse_problems_solver import GenerativeModelSolver
from embeddings.plotting.city_emission_field_plot import plot_emission_field_tensor

if __name__ == "__main__":
    dataset = TnoDatasetCollection(deterministic=True).validation_data
    dataset.disable_temporal_transforms()

    num_measurements = 500

    solver = GenerativeModelSolver(plot_loss=True, log_info=True)
    experiment = CompressedSensingExperiment(num_measurements=num_measurements)

    x = dataset[random.randint(0, len(dataset) - 1)]
    x_rec = experiment.solve_random_inverse_problem(x=x, solver=solver)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))

    vmax = 1.1 * float(torch.max(x))

    plot_emission_field_tensor(emission_field=x, ax=ax1, vmax=vmax)
    ax1.title.set_text("Original Emission Field")
    plot_emission_field_tensor(emission_field=x_rec, ax=ax2, vmax=vmax)
    ax2.title.set_text(f"Reconstructed Emission Field\nWith {num_measurements} measurements")

    plt.savefig(PlotPaths.PLOTS / "inverse_problem.png")