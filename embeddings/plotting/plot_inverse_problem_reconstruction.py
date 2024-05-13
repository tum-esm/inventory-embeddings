import random

import numpy as np
import torch
from matplotlib import pyplot as plt

from embeddings.common.log import logger
from embeddings.common.paths import ModelPaths
from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection
from embeddings.evaluation.compressed_sensing_experiment import generate_random_inverse_problem, solve_inverse_problem
from embeddings.evaluation.inverse_problems_solver import GenerativeModelSolver
from embeddings.plotting.city_emission_field_plot import plot_emission_field_tensor

if __name__ == "__main__":
    dataset = TnoDatasetCollection().test_data
    dataset.disable_temporal_transforms()

    num_measurements = 500

    solver = GenerativeModelSolver(plot_loss=True, log_info=True)

    x = dataset[random.randint(0, len(dataset) - 1)]
    inverse_problem = generate_random_inverse_problem(x=x, num_measurements=num_measurements)
    x_rec = solve_inverse_problem(solver=solver, inverse_problem=inverse_problem)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))

    vmax = 1.1 * float(torch.max(x))

    plot_emission_field_tensor(emission_field=x, ax=ax1, vmax=vmax)
    ax1.title.set_text("Original Emission Field")
    plot_emission_field_tensor(emission_field=x_rec, ax=ax2, vmax=vmax)
    ax2.title.set_text(f"Reconstructed Emission Field\nWith {num_measurements} measurements")

    logger.info(f"Reconstruction Loss: {float(np.square(np.subtract(x, x_rec)).mean())}")

    ModelPaths.VAE_LATEST_PLOTS.mkdir(exist_ok=True)
    plt.savefig(ModelPaths.VAE_LATEST_PLOTS / "inverse_reconstruction.png")
