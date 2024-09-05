import polars as pl

from src.common.constants import TNO_LAT_STEP, TNO_LON_STEP
from src.common.log import logger
from src.common.paths import OpenDataSoftPaths
from src.data_preprocessing.data_classes.city import City


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


def _are_coordinates_too_close(city_1: City, city_2: City) -> bool:
    threshold = 40  # ~km

    return all(
        [
            city_1.lat - threshold * TNO_LAT_STEP <= city_2.lat <= city_1.lat + threshold * TNO_LAT_STEP,
            city_1.lon - threshold * TNO_LON_STEP <= city_2.lon <= city_1.lon + threshold * TNO_LON_STEP,
        ],
    )


def filter_cities_from_open_data_soft_data(
    min_population_size: int,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    show_warnings: bool = False,
) -> list[City]:
    res = pl.read_csv(OpenDataSoftPaths.GEONAMES_CSV, separator=";", infer_schema_length=10000)

    filtered_cities = res.filter(
        (pl.col("Population") >= min_population_size),
        (
            pl.col("Coordinates").map_elements(
                function=lambda val: _are_coordinates_within_range(val, lat_range=lat_range, lon_range=lon_range),
                return_dtype=pl.Boolean,
            )
        ),
    )

    cities: list[City] = []

    for city_data in filtered_cities.iter_rows(named=True):
        lat, lon = _coordinates_to_float(city_data["Coordinates"])
        population = city_data["Population"]
        city = City(name=city_data["Name"], lat=lat, lon=lon, population=population)

        close_cities = [i for i in range(len(cities)) if _are_coordinates_too_close(city_1=cities[i], city_2=city)]

        if close_cities:
            city_names = ", ".join(cities[i].name for i in close_cities)
            if all(city.population >= cities[i].population for i in close_cities):
                if show_warnings:
                    logger.warning(f"Removing {city_names} in favor of {city.name}")
                for j, i in enumerate(close_cities):
                    cities.pop(i - j)
            else:
                if show_warnings:
                    logger.warning(f"Skipping {city.name} as it is too close to {city_names}")
                continue
        cities.append(city)

    return cities
