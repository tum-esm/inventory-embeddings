import polars as pl

from src.common.paths import ExperimentPath

from . import MEASUREMENTS, RELATIVE_ERROR, SNR, SOLVER, SSIM


class EvaluationDataReader:
    def __init__(self, path: ExperimentPath) -> None:
        data = pl.read_csv(path.csv_path, dtypes={"Solver": pl.String})
        self._measurements = self._extract_sorted_num_measurements(data)
        self._snr = self._extract_sorted_snr(data)
        self._data = data

    @property
    def measurements(self) -> list[int]:
        return self._measurements

    @property
    def snr_values(self) -> list[int | None]:
        return self._snr

    def _extract_sorted_num_measurements(self, data: pl.DataFrame) -> list[int]:
        return sorted([m for m, _ in data.groupby(MEASUREMENTS)])

    def _extract_sorted_snr(self, data: pl.DataFrame) -> list[int | None]:
        return sorted([snr for snr, _ in data.groupby(SNR)])

    def get_mean_relative_error_per_measurements(self, solver: str) -> list[float]:
        solver_data = self._data.filter(pl.col(SOLVER) == solver)
        result = []
        for m in self._measurements:
            measurements = solver_data.filter(pl.col(MEASUREMENTS) == m)
            result.append(measurements[RELATIVE_ERROR].mean())
        return result

    def get_mean_ssim_per_measurements(self, solver: str) -> list[float]:
        solver_data = self._data.filter(pl.col(SOLVER) == solver)
        result = []
        for m in self._measurements:
            measurements = solver_data.filter(pl.col(MEASUREMENTS) == m)
            result.append(measurements[SSIM].mean())
        return result
