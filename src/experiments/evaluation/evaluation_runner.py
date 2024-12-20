from dataclasses import dataclass
from sys import stdout

from tqdm import tqdm

from src.common.csv_writer import CsvWriter
from src.common.paths import ExperimentPath
from src.dataset.tno_dataset import TnoDataset
from src.inverse_problems.compressed_sensing_problem import SectorWiseCompressedSensingProblem
from src.inverse_problems.inverse_problems_solver import InverseProblemSolver
from src.models.common.metrics import relative_error, ssim

from . import MEASUREMENTS, RELATIVE_ERROR, SNR, SOLVER, SSIM


@dataclass
class EvaluationSettings:
    path: ExperimentPath
    measurements: list[int]
    snr: list[float | None]
    dataset: TnoDataset
    repetitions: int = 1


class EvaluationRunner:
    def __init__(self, settings: EvaluationSettings) -> None:
        self._path = settings.path
        self._measurements = settings.measurements
        self._snr = settings.snr
        self._dataset = settings.dataset
        self._repetitions = settings.repetitions
        self._num_problems = len(self._dataset) * len(self._measurements) * len(self._snr) * self._repetitions

    def run(self, solvers: dict[str, InverseProblemSolver], iterations: int = 1) -> None:
        self._path.archive()
        if iterations == 1:
            self._run_for_each_solver(solvers)
            return
        for _ in tqdm(range(iterations), desc="Run"):
            self._run_for_each_solver(solvers)

    def _run_for_each_solver(self, solvers: dict[str, InverseProblemSolver]) -> None:
        evaluation_csv = CsvWriter(self._path.csv_path)

        evaluation_csv.write_header(SOLVER, MEASUREMENTS, SNR, RELATIVE_ERROR, SSIM)

        with tqdm(total=len(solvers) * self._num_problems, desc="Evaluation", file=stdout) as bar:
            for x in self._dataset:
                for snr in self._snr:
                    for num_measurements in self._measurements:
                        inverse_problem = SectorWiseCompressedSensingProblem.generate_random_sector_wise_measurements(
                            x=x,
                            num_measurements=num_measurements,
                            snr=snr,
                        )
                        for name, solver in solvers.items():
                            for _ in range(self._repetitions):
                                x_rec = inverse_problem.solve(solver)
                                relative = relative_error(x=x, x_hat=x_rec).item()
                                s = ssim(x=x, x_hat=x_rec).item()
                                evaluation_csv.write_row(name, num_measurements, snr, relative, s)
                                bar.update()
