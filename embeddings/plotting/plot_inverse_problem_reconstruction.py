import random

import torch
from matplotlib import pyplot as plt
from matplotlib.axis import Axis
from torch import Tensor

from embeddings.common.gnfr_sector import GnfrSector
from embeddings.common.log import logger
from embeddings.common.paths import ModelPathsCreator, PlotPaths
from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection
from embeddings.evaluation.compressed_sensing_experiment import generate_random_inverse_problem, solve_inverse_problem
from embeddings.evaluation.inverse_problems_solver import (
    DctLassoSolver,
    DwtLassoSolver,
    GenerativeModelSolver,
    LassoSolver,
)
from embeddings.models.common.metrics import mse, ssim
from embeddings.plotting.city_emission_field_plot import plot_emission_field_tensor

SECTORS_TO_PLOT = [GnfrSector.B, GnfrSector.C, GnfrSector.F2]


def plot_reconstruction(axes_: tuple[tuple[Axis]], field: Tensor, vmax_: float, title: str, index_: int) -> None:
    ax = axes_[0][index_]
    plot_emission_field_tensor(emission_field=field, ax=ax, vmax=vmax_)
    ax.title.set_text(title)

    for i, sector in enumerate(SECTORS_TO_PLOT):
        ax = axes_[i + 1][index_]
        plot_emission_field_tensor(emission_field=field, ax=ax, vmax=vmax_, sector=sector)
        ax.title.set_text(f"{sector}")


if __name__ == "__main__":
    dataset = TnoDatasetCollection().test_data
    dataset.disable_temporal_transforms()

    num_measurements = 500

    solvers = {
        "VAE 128": GenerativeModelSolver(path_to_model=ModelPathsCreator.get_vae_model("128")),
        "VAE 256": GenerativeModelSolver(path_to_model=ModelPathsCreator.get_vae_model("256")),
        "Lasso": LassoSolver(),
        "Lasso (DWT)": DwtLassoSolver(),
        "Lasso (DCT)": DctLassoSolver(),
    }

    x = dataset[random.randint(0, len(dataset) - 1)]
    inverse_problem = generate_random_inverse_problem(x=x, num_measurements=num_measurements)

    fig, axes = plt.subplots(
        len(SECTORS_TO_PLOT) + 1,
        len(solvers) + 1,
        figsize=(5 + 5 * len(solvers), 5 + 5 * len(SECTORS_TO_PLOT)),
    )
    vmax = 1.1 * float(torch.max(x))

    plot_reconstruction(
        axes_=axes,
        field=x,
        index_=0,
        vmax_=vmax,
        title="Original Emission Field",
    )

    for index, (solver_name, solver) in enumerate(solvers.items()):
        x_rec = solve_inverse_problem(solver=solver, inverse_problem=inverse_problem)

        plot_reconstruction(
            axes_=axes,
            field=x_rec,
            index_=index + 1,
            vmax_=vmax,
            title=f"Reconstructed Emission Field\nWith {num_measurements} measurements\n{solver_name}",
        )

        logger.info(f"MSE {solver_name}: {mse(x=x, x_hat=x_rec)}")
        logger.info(f"SSIM {solver_name}: {ssim(x=x, x_hat=x_rec)}")

    plt.savefig(PlotPaths.PLOTS / "inverse_reconstruction.png")
