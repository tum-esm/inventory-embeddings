import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from embeddings.common.paths import ExperimentPaths


def _get_mean_and_confidence_tuples(pl_values: pl.DataFrame, metric: str) -> dict[int, tuple[float, float]]:
    values_dict = {}
    for num_measurements, group in pl_values.group_by(["Measurements"]):
        values = group.get_column(metric).to_numpy()
        mean = np.mean(values)
        std_error = np.std(values, ddof=1) / np.sqrt(len(values))
        confidence_interval = 1.96 * std_error  # 95% confidence interval
        values_dict[num_measurements[0]] = (mean, confidence_interval)
        values_dict = dict(sorted(values_dict.items()))
    return values_dict


if __name__ == "__main__":
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))

    data_frames = [pl.read_csv(evaluation_csv) for evaluation_csv in ExperimentPaths.EVALUATION_LATEST.glob("*.csv")]
    evaluation = pl.concat(data_frames)

    for solver, group in evaluation.group_by("Solver"):
        mse_values = _get_mean_and_confidence_tuples(pl_values=group, metric="MSE")
        ssim_values = _get_mean_and_confidence_tuples(pl_values=group, metric="SSIM")

        measurements = list(mse_values.keys())

        mean_mse = [value[0] for value in mse_values.values()]
        confidence_mse = [value[1] / 2 for value in mse_values.values()]

        mean_ssim = [value[0] for value in ssim_values.values()]
        confidence_ssim = [value[1] / 2 for value in ssim_values.values()]

        ax1.errorbar(x=measurements, y=mean_mse, yerr=confidence_mse, label=solver)
        ax2.errorbar(x=measurements, y=mean_ssim, yerr=confidence_ssim, label=solver)

    ax1.set_xscale("log")
    ax1.set_xticks(ticks=measurements, labels=[str(m) for m in measurements], rotation=90)
    ax1.set_xlim([measurements[0], measurements[-1]])
    ax1.legend()

    ax2.set_xscale("log")
    ax2.set_xticks(ticks=measurements, labels=[str(m) for m in measurements], rotation=90)
    ax2.set_xlim([measurements[0], measurements[-1]])
    ax2.legend()

    plt.savefig(ExperimentPaths.EVALUATION_LATEST / "evaluation.png")
