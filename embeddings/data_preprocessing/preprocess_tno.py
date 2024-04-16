import argparse

from embeddings.common.paths import TnoPaths
from embeddings.data_preprocessing.tno_preprocessor import TnoPreprocessor, TnoPreprocessorOptions


def preprocess() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    TnoPaths.BY_CITY_2015_CSV.parent.mkdir(exist_ok=True)

    grid_width_help = "width of grid around city center"
    grid_height_help = "height of grid around city center"
    population_help = "minimum required population of cities"

    parser.add_argument("-gw", "--width", metavar="W", default=61, type=int, help=grid_width_help)
    parser.add_argument("-gh", "--height", metavar="H", default=61, type=int, help=grid_height_help)
    parser.add_argument("-p", "--population", metavar="N", default=500_000, type=int, help=population_help)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    options = TnoPreprocessorOptions(
        grid_width=args.width,
        grid_height=args.height,
        min_population_size=args.population,
        show_warnings=args.verbose,
    )

    preprocessor = TnoPreprocessor(options=options)
    preprocessor.preprocess(
        tno_csv=TnoPaths.HIGH_RES_2015_CSV,
        out_csv=TnoPaths.BY_CITY_2015_CSV,
    )
