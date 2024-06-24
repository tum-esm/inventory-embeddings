import random

import torch
from matplotlib import pyplot as plt
from matplotlib.axis import Axis
from torch import Tensor

from embeddings.common.gnfr_sector import GnfrSector
from embeddings.common.log import logger
from embeddings.common.paths import PlotPaths
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


def plot_field(axes_: tuple[tuple[Axis]], field: Tensor, vmax_: float, title: str, index_: int) -> None:
    ax = axes_[0][index_]
    plot_emission_field_tensor(emission_field=field, ax=ax, vmax=vmax_)
    ax.title.set_text(title)

    for i, sector in enumerate(SECTORS_TO_PLOT):
        ax = axes_[i + 1][index_]
        plot_emission_field_tensor(emission_field=field, ax=ax, vmax=vmax_, sector=sector)
        ax.title.set_text(f"{sector}")


if __name__ == "__main__":
    SNR = 100

    city = "Munich"
    dataset = TnoDatasetCollection().get_case_study_data(city=city, year=2018)
    dataset.disable_temporal_transforms()

    num_measurements = 1000

    solvers = {
        "VAE 256": GenerativeModelSolver.from_vae_model_name("256"),
        "VAE 512": GenerativeModelSolver.from_vae_model_name("512"),
        "VAE 1024": GenerativeModelSolver.from_vae_model_name("1024"),
        "VAE 2048": GenerativeModelSolver.from_vae_model_name("2048"),
        "VAE 2048 Munich": GenerativeModelSolver.from_vae_model_name("2048_fine_tuned_on_munich"),
        "Lasso": LassoSolver(),
        "Lasso (DWT)": DwtLassoSolver(),
        "Lasso (DCT)": DctLassoSolver(),
    }

    x = dataset[random.randint(0, len(dataset) - 1)]
    inverse_problem = generate_random_inverse_problem(x=x, num_measurements=num_measurements, signal_to_noise_ratio=SNR)

    fig_1, axes = plt.subplots(
        len(SECTORS_TO_PLOT) + 1,
        len(solvers) + 1,
        figsize=(5 + 5 * len(solvers), 5 + 5 * len(SECTORS_TO_PLOT)),
    )
    vmax = 1.1 * float(torch.max(x))

    fig_2, axes_difference = plt.subplots(
        len(SECTORS_TO_PLOT) + 1,
        len(solvers),
        figsize=(5 * len(solvers), 5 + 5 * len(SECTORS_TO_PLOT)),
    )

    plot_field(
        axes_=axes,
        field=x,
        index_=0,
        vmax_=vmax,
        title="Original Emission Field",
    )

    for index, (solver_name, solver) in enumerate(solvers.items()):
        x_rec = solve_inverse_problem(solver=solver, inverse_problem=inverse_problem)

        plt.figure(1)

        plot_field(
            axes_=axes,
            field=x_rec,
            index_=index + 1,
            vmax_=vmax,
            title=f"Reconstructed Emission Field\nWith {num_measurements} measurements\n{solver_name}",
        )

        plt.figure(2)

        plot_field(
            axes_=axes_difference,
            field=x_rec - x,
            index_=index,
            vmax_=vmax,
            title=f"Difference Map \n{solver_name}",
        )

        logger.info(f"MSE {solver_name}: {mse(x=x, x_hat=x_rec)}")
        logger.info(f"SSIM {solver_name}: {ssim(x=x, x_hat=x_rec)}")

    path = PlotPaths.CASE_STUDY_PLOT / city.lower()
    path.mkdir(exist_ok=True)
    plt.figure(1)
    plt.savefig(path / f"cs_reconstruction_snr_{SNR}_{num_measurements!s}.png")
    plt.figure(2)
    plt.savefig(path / f"cs_reconstruction_snr_{SNR}_{num_measurements!s}_diff.png")
