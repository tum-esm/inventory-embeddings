import random

import torch
from matplotlib import pyplot as plt
from torch import Tensor

from embeddings.common.gnfr_sector import NUM_GNFR_SECTORS, GnfrSector
from embeddings.common.paths import PlotPaths
from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection
from embeddings.evaluation.inverse_problems_solver import GenerativeModelSolver, InverseProblemSolver
from embeddings.plotting.city_emission_field_plot import plot_emission_field_tensor


class CompressedSensingExperiment:
    def __init__(self, num_measurements: int) -> None:
        self._dataset = TnoDatasetCollection(deterministic=True).training_data
        self._num_measurements = num_measurements
        self._emission_field_width = TnoDatasetCollection.CROPPED_WIDTH
        self._emission_field_height = TnoDatasetCollection.CROPPED_HEIGHT
        self._emission_field_depth = NUM_GNFR_SECTORS
        self._emission_field_size = self._emission_field_depth * self._emission_field_width * self._emission_field_width

    def _generate_random_forward_model(self) -> Tensor:
        return torch.randn((self._num_measurements, self._emission_field_size))

    def _sample_random_emission_field(self) -> Tensor:
        random_index = random.randint(0, len(self._dataset) - 1)
        return self._dataset[random_index]

    def _vectorize(self, x: Tensor) -> Tensor:
        return x.view(self._emission_field_size)

    def _un_vectorize(self, x: Tensor) -> Tensor:
        return x.view(self._emission_field_depth, self._emission_field_height, self._emission_field_height)

    def run_inverse_problem(self, solver: InverseProblemSolver) -> tuple[Tensor, Tensor]:
        A = self._generate_random_forward_model()  # noqa: N806
        x = self._sample_random_emission_field()
        x_vectorized = self._vectorize(x)
        y = A @ x_vectorized
        x_rec_vectorized = solver.solve(y=y, A=A)
        x_rec = self._un_vectorize(x_rec_vectorized)
        return x, x_rec


if __name__ == "__main__":
    num_measurements = 250

    solver = GenerativeModelSolver()
    experiment = CompressedSensingExperiment(num_measurements=250)

    x, x_rec = experiment.run_inverse_problem(solver=solver)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))

    sector = GnfrSector.F1

    vmax = 1.1 * float(torch.max(x))
    vmax_sector = 1.1 * float(torch.max(x[sector.to_index(), :, :]))

    plot_emission_field_tensor(emission_field=x, ax=ax1, vmax=vmax)
    ax1.title.set_text("Original Emission Field")
    plot_emission_field_tensor(emission_field=x_rec, ax=ax2, vmax=vmax)
    ax2.title.set_text(f"Reconstructed Emission Field\nWith {num_measurements} measurements")

    plt.savefig(PlotPaths.PLOTS / "inverse_problem.png")
