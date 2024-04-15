from embeddings.common.paths import TnoPaths
from embeddings.data_preprocessing.tno_preprocessor import TnoPreprocessor, TnoPreprocessorOptions


def preprocess() -> None:
    TnoPaths.BY_CITY_2015_CSV.parent.mkdir(exist_ok=True)

    preprocessor = TnoPreprocessor(options=TnoPreprocessorOptions())
    preprocessor.preprocess(
        tno_csv=TnoPaths.HIGH_RES_2015_CSV,
        out_csv=TnoPaths.BY_CITY_2015_CSV,
    )
