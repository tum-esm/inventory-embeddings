import argparse

from src.common.log import logger
from src.common.paths import TnoPaths
from src.data_preprocessing.tno_preprocessor import TnoPreprocessor, TnoPreprocessorOptions


def preprocess() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    grid_width_help = "width of grid around city center"
    grid_height_help = "height of grid around city center"
    population_help = "minimum required population of cities"

    parser.add_argument("-gw", "--width", metavar="W", default=51, type=int, help=grid_width_help)
    parser.add_argument("-gh", "--height", metavar="H", default=51, type=int, help=grid_height_help)
    parser.add_argument("-p", "--population", metavar="N", default=100_000, type=int, help=population_help)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    TnoPaths.BY_CITY.mkdir(exist_ok=True)

    options = TnoPreprocessorOptions(
        grid_width=args.width,
        grid_height=args.height,
        min_population_size=args.population,
        show_warnings=args.verbose,
    )

    preprocessor = TnoPreprocessor(options=options)

    logger.info(f"Preprocessing TNO data at '{TnoPaths.HIGH_RES_2015_CSV}'")
    preprocessor.preprocess(
        tno_csv=TnoPaths.HIGH_RES_2015_CSV,
        out_csv=TnoPaths.BY_CITY_2015_CSV,
    )

    logger.info(f"Preprocessing TNO data at '{TnoPaths.HIGH_RES_2018_CSV}'")
    preprocessor.preprocess(
        tno_csv=TnoPaths.HIGH_RES_2018_CSV,
        out_csv=TnoPaths.BY_CITY_2018_CSV,
    )
