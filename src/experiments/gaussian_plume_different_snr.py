import argparse
from sys import stdout

import numpy as np
from torch import Tensor
from tqdm import tqdm

from src.common.csv_writer import CsvWriter
from src.common.paths import ExperimentPath
from src.dataset.tno_dataset_collection import TnoDatasetCollection
from src.experiments.hyper_parameters import LAMBDA_PER_SNR
from src.inverse_problems.compressed_sensing_problem import TotalEmissionsCompressedSensingExperiment
from src.inverse_problems.footprints.footprint_loader import load_gaussian_plume_footprint
from src.inverse_problems.inverse_problems_solver import (
    BasisPursuitSolver,
    InverseProblemSolver,
    LeastSquaresSolver,
    SparseGenerativeModelSolver,
    SparsityTransform,
)
from src.models.common.metrics import relative_error, ssim

SNR_DB_TO_RUN = [5, 10, 15, 20, 25, 30, 35, 40]
NUM_MEASUREMENT_STATIONS = 30
DATA_YEAR = 2018
ITERATIONS_PER_PROBLEM = 3

SOLVER_NOT_FOUND_ERROR = "Solver could not be found!"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    city_help = "City to run the experiment on."
    solver_help = "Solver to use. Options: LS, DCT, DWT, 256, 512, 1024, 2048, 256_munich, ..."

    parser.add_argument("-c", "--city", metavar="C", type=str, help=city_help, required=True)
    parser.add_argument("-s", "--solver", metavar="S", help=solver_help, type=str, required=True)

    args = parser.parse_args()

    solver: InverseProblemSolver | None = None

    if args.solver == "LS":
        solver = LeastSquaresSolver()
    elif args.solver == "DCT":
        solver = BasisPursuitSolver(transform=SparsityTransform.DCT)
    elif args.solver == "DWT":
        solver = BasisPursuitSolver(transform=SparsityTransform.DWT)
    else:
        solver = SparseGenerativeModelSolver.from_vae_model_name(args.solver)

    if solver is None:
        raise ValueError(SOLVER_NOT_FOUND_ERROR)

    dataset_collection = TnoDatasetCollection()
    city_data = dataset_collection.get_single_case_study_city_emission_field(city=args.city, year=DATA_YEAR)

    x_a = city_data.co2_ff_area_sources_tensor.sum(0)

    sensing_matrix = Tensor(load_gaussian_plume_footprint(NUM_MEASUREMENT_STATIONS))

    def create_inverse_problem(snr: float) -> TotalEmissionsCompressedSensingExperiment:
        return TotalEmissionsCompressedSensingExperiment.generate_from_sensing_matrix(
            x=x_a,
            sensing_matrix=sensing_matrix,
            snr=snr,
        )

    experiment_path = ExperimentPath("gaussian_plume_snr")
    csv_path = experiment_path.path / args.city.lower() / f"{args.solver.lower()}.csv"
    csv_path.unlink(missing_ok=True)
    csv_path.parent.mkdir(exist_ok=True, parents=True)

    csv_writer = CsvWriter(csv_path)

    csv_writer.write_header("snr_db", "relative_error", "ssim")

    for snr_db in tqdm(SNR_DB_TO_RUN, desc="Experiment", file=stdout):
        relative_error_values = np.zeros(ITERATIONS_PER_PROBLEM)
        ssim_values = np.zeros(ITERATIONS_PER_PROBLEM)

        for i in range(ITERATIONS_PER_PROBLEM):
            problem = create_inverse_problem(snr=10 ** (snr_db / 10))
            if isinstance(solver, SparseGenerativeModelSolver):
                x_rec = problem.solve(solver, lambda_=LAMBDA_PER_SNR[args.city][args.solver][snr_db])
            else:
                x_rec = problem.solve(solver)
            relative_error_values[i] = relative_error(x=x_a, x_hat=x_rec).item()
            ssim_values[i] = ssim(x=x_a, x_hat=x_rec).item()

        csv_writer.write_row(
            snr_db,
            relative_error_values.mean(),
            ssim_values.mean(),
        )
