from embeddings.common.paths import TnoPaths
from embeddings.data_preprocessing.filter_tno_data_by_cities import filter_tno_data_by_cities


def preprocess() -> None:
    TnoPaths.BY_CITY_2015_CSV.parent.mkdir(exist_ok=True)

    filter_tno_data_by_cities(
        tno_data_csv=TnoPaths.HIGH_RES_2015_CSV,
        out_csv=TnoPaths.BY_CITY_2015_CSV,
    )
