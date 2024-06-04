from datetime import UTC, datetime
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).parent.parent.parent
_DATA = _REPOSITORY_ROOT / "data"
_SAVES = _REPOSITORY_ROOT / "saves"


def _archive_dir(path: Path) -> None:
    if path.exists():
        now = datetime.now(UTC)
        timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
        path.replace(target=path.parent / f"archived_{timestamp}")
    path.mkdir(exist_ok=True, parents=True)


class TnoPaths:
    _HIGH_RES = _DATA / "TNO-GHGco-1km"

    _HIGH_RES_2015 = _HIGH_RES / "TNO_highres_2015"
    HIGH_RES_2015_CSV = _HIGH_RES_2015 / "TNO_GHGco_2015_highres_v1_1.csv"

    _HIGH_RES_2018 = _HIGH_RES / "TNO_highres_2018"
    HIGH_RES_2018_CSV = _HIGH_RES_2018 / "TNO_GHGco_v4_0_highres_year2018.csv"

    BY_CITY = _HIGH_RES / "TNO_by_city"
    BY_CITY_2015_CSV = BY_CITY / "TNO_GHGco_2015_highres_by_city.csv"
    BY_CITY_2018_CSV = BY_CITY / "TNO_GHGco_2018_highres_by_city.csv"

    _TIME_PROFILES = _HIGH_RES / "TNO_timeprofiles"
    HOUR_TIME_PROFILE = _TIME_PROFILES / "timeprofiles-hour-in-day_GNFR.csv"
    DAY_TIME_PROFILE = _TIME_PROFILES / "timeprofiles-day-in-week_GNFR.csv"
    MONTH_TIME_PROFILE = _TIME_PROFILES / "timeprofiles-month-in-year_GNFR.csv"


class OpenDataSoftPaths:
    _OPEN_DATA_SOFT = _DATA / "OpenDataSoft"
    GEONAMES_CSV = _OPEN_DATA_SOFT / "geonames-all-cities-with-a-population-1000.csv"


class PlotPaths:
    PLOTS = _SAVES / "plots"


class ModelPaths:
    _MODELS = _SAVES / "models"
    _VAE_MODELS = _MODELS / "vae"

    VAE_LATEST = _VAE_MODELS / "latest"

    VAE_LATEST_CHECKPOINTS = VAE_LATEST / "checkpoints"
    VAE_LATEST_PLOTS = VAE_LATEST / "plots"

    @classmethod
    def archive_latest_vae_model(cls) -> None:
        _archive_dir(cls.VAE_LATEST)


class ExperimentPaths:
    _EXPERIMENTS = _SAVES / "experiments"
    _EVALUATIONS = _EXPERIMENTS / "evaluations"

    EVALUATION_LATEST = _EVALUATIONS / "latest"

    @classmethod
    def archive_latest_evaluation(cls) -> None:
        _archive_dir(cls.EVALUATION_LATEST)
