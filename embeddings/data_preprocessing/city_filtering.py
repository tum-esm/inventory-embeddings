import polars as pl

from embeddings.common.paths import OpenDataSoftPaths
from embeddings.data_preprocessing.data_classes.city import City


def _coordinates_to_float(coordinates: str) -> tuple[float, float]:
    lat, lon = coordinates.split(",")
    return float(lat), float(lon)


def _are_coordinates_within_range(
    coordinates: str,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
) -> bool:
    lat, lon = _coordinates_to_float(coordinates)
    return lat_range[0] <= lat <= lat_range[1] and lon_range[0] <= lon <= lon_range[1]


def filter_cities_from_open_data_soft_data(
    min_population_size: int,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
) -> list[City]:
    res = pl.read_csv(OpenDataSoftPaths.GEONAMES_CSV, separator=";", infer_schema_length=10000)

    filtered_cities = res.filter(
        (pl.col("Population") >= min_population_size),
        (
            pl.col("Coordinates").map_elements(
                function=lambda val: _are_coordinates_within_range(val, lat_range=lat_range, lon_range=lon_range),
                return_dtype=bool,
            )
        ),
    )

    cities = []

    for city in filtered_cities.iter_rows(named=True):
        lat, lon = _coordinates_to_float(city["Coordinates"])
        cities.append(City(name=city["Name"], lat=lat, lon=lon))

    return cities
