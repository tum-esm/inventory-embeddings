from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from src.common.csv_writer import CsvWriter
from src.common.paths import ExperimentPaths
from src.dataset.tno_dataset import TnoDataset
from src.inverse_problems.compressed_sensing_problem import SectorWiseCompressedSensingProblem
from src.inverse_problems.inverse_problems_solver import InverseProblemSolver
from src.models.common.metrics import mse, relative_error, ssim


@dataclass
class EvaluationSettings:
    base_path: Path
    measurements: list[int]
    snr: list[int]
    dataset: TnoDataset


class Evaluation:
    def __init__(self, settings: EvaluationSettings) -> None:
        self._base_path = settings.base_path
        self._base_path.mkdir(exist_ok=True, parents=True)
        self._measurements = settings.measurements
        self._snr = settings.snr
        self._dataset = settings.dataset
        self._number_of_iterations = len(self._dataset) * len(self._measurements) * len(self._snr)

    def run(self, solvers: dict[str, InverseProblemSolver], iterations: int = 1) -> None:
        if iterations == 1:
            self._run_for_each_solver(solvers)
            return
        for _ in tqdm(range(iterations), desc="Run"):
            self._run_for_each_solver(solvers)

    def _run_for_each_solver(self, solvers: dict[str, InverseProblemSolver]) -> None:
        evaluation_csv_path = ExperimentPaths.get_next_evaluation_csv(path=self._base_path)
        evaluation_csv = CsvWriter(evaluation_csv_path)

        evaluation_csv.write_header("Solver", "Measurements", "SNR", "MSE", "Relative Error", "SSIM")

        with tqdm(total=self._number_of_iterations, leave=False, desc="Evaluation") as bar:
            for num_measurements in self._measurements:
                for snr in self._snr:
                    for x in self._dataset:
                        inverse_problem = SectorWiseCompressedSensingProblem.generate_random_sector_wise_measurements(
                            x=x,
                            num_measurements=num_measurements,
                            snr=snr,
                        )
                        for name, solver in solvers.items():
                            x_rec = inverse_problem.solve(solver)
                            mean_squared = mse(x=x, x_hat=x_rec).item()
                            relative = relative_error(x=x, x_hat=x_rec).item()
                            s = ssim(x=x, x_hat=x_rec).item()
                            evaluation_csv.write_row(name, num_measurements, snr, mean_squared, relative, s)
                        bar.update()
