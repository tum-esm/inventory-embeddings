import random

import torch
from matplotlib import pyplot as plt

from embeddings.common.log import logger
from embeddings.common.paths import ModelPaths
from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection
from embeddings.evaluation.compressed_sensing_experiment import generate_random_inverse_problem, solve_inverse_problem
from embeddings.evaluation.inverse_problems_solver import DwtLassoSolver, GenerativeModelSolver, LassoSolver
from embeddings.models.common.metrics import mse, ssim
from embeddings.plotting.city_emission_field_plot import plot_emission_field_tensor

if __name__ == "__main__":
    dataset = TnoDatasetCollection().test_data
    dataset.disable_temporal_transforms()

    num_measurements = 500

    solvers = {
        "Generative AI": GenerativeModelSolver(plot_loss=True, log_info=True),
        "Lasso": LassoSolver(),
        "Lasso + DWT": DwtLassoSolver(),
    }

    x = dataset[random.randint(0, len(dataset) - 1)]
    inverse_problem = generate_random_inverse_problem(x=x, num_measurements=num_measurements)

    fig, axes = plt.subplots(1, len(solvers) + 1, figsize=(5 + 5 * len(solvers), 8))
    vmax = 1.1 * float(torch.max(x))
    plot_emission_field_tensor(emission_field=x, ax=axes[0], vmax=vmax)
    axes[0].title.set_text("Original Emission Field")

    for index, (solver_name, solver) in enumerate(solvers.items()):
        x_rec = solve_inverse_problem(solver=solver, inverse_problem=inverse_problem)

        ax = axes[index + 1]
        plot_emission_field_tensor(emission_field=x_rec, ax=ax, vmax=vmax)
        ax.title.set_text(f"Reconstructed Emission Field\nWith {num_measurements} measurements\n{solver_name}")

        logger.info(f"MSE {solver_name}: {mse(x=x, x_hat=x_rec)}")
        logger.info(f"SSIM {solver_name}: {ssim(x=x, x_hat=x_rec)}")

    ModelPaths.VAE_LATEST_PLOTS.mkdir(exist_ok=True)
    plt.savefig(ModelPaths.VAE_LATEST_PLOTS / "inverse_reconstruction.png")
