from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).parent.parent.parent
_DATA = _REPOSITORY_ROOT / "data"

PLOTS = _REPOSITORY_ROOT / "plots"


class TnoPaths:
    _TNO_HIGH_RES = _DATA / "TNO-GHGco-1km"

    _TNO_HIGH_RES_2015 = _TNO_HIGH_RES / "TNO_highres_2015"
    TNO_HIGH_RES_2015_CSV = _TNO_HIGH_RES_2015 / "TNO_GHGco_2015_highres_v1_1.csv"

    _TNO_BY_CITY = _TNO_HIGH_RES / "TNO-by-city"
    TNO_BY_CITY_2015_CSV = _TNO_BY_CITY / "TNO_GHGco_2015_highres_v1_1_by_city.csv"


class OpenDataSoftPaths:
    _OPEN_DATA_SOFT = _DATA / "OpenDataSoft"
    OPEN_DATA_SOFT_GEONAMES_CSV = _OPEN_DATA_SOFT / "geonames-all-cities-with-a-population-1000.csv"
