import random

import torch
from matplotlib import pyplot as plt
from matplotlib.axis import Axis
from torch import Tensor

from embeddings.common.gnfr_sector import GnfrSector
from embeddings.common.log import logger
from embeddings.common.paths import PlotPaths
from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection
from embeddings.inverse_problems.compressed_sensing_problem import SectorWiseCompressedSensingProblem
from embeddings.inverse_problems.inverse_problems_solver import (
    GenerativeModelSolver,
    LassoSolver,
    SparsityTransform,
)
from embeddings.models.common.metrics import mse, relative_error, ssim
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
    SNR = 0

    city = "Munich"
    dataset = TnoDatasetCollection().get_case_study_data(city=city, year=2018)
    dataset.disable_temporal_transforms()

    num_measurements = 5000

    solvers = {
        "VAE 2048": GenerativeModelSolver.from_vae_model_name("2048"),
        "VAE 2048 Munich": GenerativeModelSolver.from_vae_model_name("2048_munich"),
        "Lasso": LassoSolver(),
        "Lasso (DCT)": LassoSolver(transform=SparsityTransform.DCT),
        "Lasso (DWT)": LassoSolver(transform=SparsityTransform.DWT),
    }

    x = dataset[random.randint(0, len(dataset) - 1)]
    compressed_sensing_problem = SectorWiseCompressedSensingProblem.generate_random_sector_wise_measurements(
        x=x,
        num_measurements=num_measurements,
        snr=SNR,
    )

    fig, axes = plt.subplots(
        len(SECTORS_TO_PLOT) + 1,
        len(solvers) + 1,
        figsize=(5 + 5 * len(solvers), 5 + 5 * len(SECTORS_TO_PLOT)),
    )
    vmax = 1.1 * float(torch.max(x))

    plot_field(
        axes_=axes,
        field=x,
        index_=0,
        vmax_=vmax,
        title="Original Emission Field",
    )

    for index, (solver_name, solver) in enumerate(solvers.items()):
        x_rec = compressed_sensing_problem.solve(solver)

        plot_field(
            axes_=axes,
            field=x_rec,
            index_=index + 1,
            vmax_=vmax,
            title=f"Reconstructed Emission Field\nWith {num_measurements} measurements\n{solver_name}",
        )

        result_string = f"Results for solver {solver_name}"
        result_string += f"\n\t\t\tMSE {solver_name}: {mse(x=x, x_hat=x_rec)}"
        result_string += f"\n\t\t\tSSIM {solver_name}: {ssim(x=x, x_hat=x_rec)}"
        result_string += f"\n\t\t\tRelative Error {solver_name}: {100 * relative_error(x=x, x_hat=x_rec):.2f}%"
        logger.info(result_string)

    path = PlotPaths.CASE_STUDY_PLOT / city.lower()
    path.mkdir(exist_ok=True, parents=True)
    plt.savefig(path / f"cs_reconstruction_snr_{SNR}_{num_measurements!s}.png")
