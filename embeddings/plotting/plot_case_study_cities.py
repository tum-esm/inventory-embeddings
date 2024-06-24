import matplotlib.pyplot as plt

from embeddings.common.gnfr_sector import GnfrSector
from embeddings.common.paths import PlotPaths
from embeddings.dataset.emission_field_transforms import RandomSparseEmittersTransform
from embeddings.dataset.tno_dataset_collection import TnoDatasetCollection
from embeddings.plotting.city_emission_field_plot import plot_emission_field

if __name__ == "__main__":
    dataset_collection = TnoDatasetCollection()

    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(20, 12))

    city = "Munich"
    data = dataset_collection.get_case_study_data(city, year=2015)
    data.disable_temporal_transforms()
    data.add_sampling_transform(RandomSparseEmittersTransform(lam=100))

    emission_field = data.get_city_emission_field(0, apply_sampling_transforms=False)
    vmax = 1.1 * emission_field.co2_ff_field.max()

    plot_emission_field(emission_field=emission_field, ax=ax1, vmax=vmax)
    plot_emission_field(emission_field=emission_field, ax=ax2, sector=GnfrSector.B, vmax=vmax)
    plot_emission_field(emission_field=emission_field, ax=ax3, sector=GnfrSector.C, vmax=vmax)
    plot_emission_field(emission_field=emission_field, ax=ax4, sector=GnfrSector.F2, vmax=vmax)

    emission_field_transformed = data.get_city_emission_field(0, apply_sampling_transforms=True)
    vmax_transformed = 1.1 * emission_field_transformed.co2_ff_field.max()

    plot_emission_field(emission_field=emission_field_transformed, ax=ax5, vmax=vmax)
    plot_emission_field(emission_field=emission_field_transformed, ax=ax6, sector=GnfrSector.B, vmax=vmax)
    plot_emission_field(emission_field=emission_field_transformed, ax=ax7, sector=GnfrSector.C, vmax=vmax)
    plot_emission_field(emission_field=emission_field_transformed, ax=ax8, sector=GnfrSector.F2, vmax=vmax)

    PlotPaths.CASE_STUDY_PLOT.mkdir(exist_ok=True)
    plt.savefig(PlotPaths.CASE_STUDY_PLOT / f"{city}_plot.png")
