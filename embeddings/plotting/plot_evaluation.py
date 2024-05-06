import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from embeddings.common.paths import ExperimentPaths, PlotPaths

if __name__ == "__main__":
    data_frames = [pl.read_csv(evaluation_csv) for evaluation_csv in ExperimentPaths.LATEST_EVALUATION.iterdir()]
    evaluation = pl.concat(data_frames)

    mse_per_measurements = {}

    for num_measurements, group in evaluation.group_by(["Measurements"]):
        mse_values = group.get_column("MSE").to_numpy()
        mean_mse = np.mean(mse_values)
        std_error = np.std(mse_values, ddof=1) / np.sqrt(len(mse_values))
        confidence_interval = 1.96 * std_error  # 95% confidence interval
        mse_per_measurements[num_measurements[0]] = (mean_mse, confidence_interval)

    mse_per_measurements_sorted = dict(sorted(mse_per_measurements.items()))

    measurements = list(mse_per_measurements_sorted.keys())
    mean_mse = [value[0] for value in mse_per_measurements_sorted.values()]
    confidence_mse = [value[1] / 2 for value in mse_per_measurements_sorted.values()]

    plt.errorbar(x=measurements, y=mean_mse, yerr=confidence_mse)
    plt.xscale("log")

    plt.savefig(PlotPaths.PLOTS / "measurements.png")
