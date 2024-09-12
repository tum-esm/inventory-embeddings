import string

import netCDF4
import numpy as np
from skimage.transform import resize

from src.common.paths import FootprintPaths

_MAX_NUM_MEASUREMENT_STATIONS = 79
_MAX_NUM_MEASUREMENT_STATIONS_EXCEEDED_ERROR = f"More than {_MAX_NUM_MEASUREMENT_STATIONS} are not allowed."

_NUM_MEASUREMENTS_PER_STATION = 50

_MEASUREMENT_SHAPE = (32, 32)
_MEASUREMENT_SHAPE_FLAT = _MEASUREMENT_SHAPE[0] * _MEASUREMENT_SHAPE[1]


def load_gaussian_plume_footprint(num_stations: int) -> np.ndarray:
    num_measurements = num_stations * _NUM_MEASUREMENTS_PER_STATION
    sensing_matrix = np.zeros((num_measurements, _MEASUREMENT_SHAPE_FLAT))

    measurement_stations = _get_name_of_measurement_stations(num_stations)

    path = FootprintPaths.SYNTHETIC_FOOTPRINTS_NC
    with netCDF4.Dataset(path, mode="r") as dataset:
        for station_index, station in enumerate(measurement_stations):
            station_data = dataset[station][:]
            for i in range(station_data.shape[2]):
                measurement = station_data[:, :, i]
                measurement_resized = resize(measurement, _MEASUREMENT_SHAPE, anti_aliasing=True)
                sensing_matrix[station_index * _NUM_MEASUREMENTS_PER_STATION + i] = measurement_resized.reshape(
                    _MEASUREMENT_SHAPE_FLAT,
                )

    return sensing_matrix


def _get_name_of_measurement_stations(num_stations: int) -> list[str]:
    if num_stations > _MAX_NUM_MEASUREMENT_STATIONS:
        raise ValueError(_MAX_NUM_MEASUREMENT_STATIONS_EXCEEDED_ERROR)
    single_letters = list(string.ascii_uppercase)
    return [a + b for a in ["", *single_letters] for b in single_letters][:num_stations]
