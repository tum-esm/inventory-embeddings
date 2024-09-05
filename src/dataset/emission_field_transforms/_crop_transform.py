import random

from src.dataset.city_emission_field import CityEmissionField

from ._emission_field_transform import EmissionFieldTransform


def _crop_emission_field(
    emission_field: CityEmissionField,
    start_x: int,
    start_y: int,
    width: int,
    height: int,
) -> None:
    end_x = start_x + width
    end_y = start_y + height

    if any([start_x < 0, start_y < 0, end_x > emission_field.width, end_y > emission_field.height]):
        value_error = "Tried to crop beyond field border!"
        raise ValueError(value_error)

    emission_field.co2_ff_field = emission_field.co2_ff_field[:, start_y:end_y, start_x:end_x]
    emission_field.lat_lon_array = emission_field.lat_lon_array[start_x:end_x, start_y:end_y, :]

    assert emission_field.width == width, f"Expected width: {width}. Got: {emission_field.width}"
    assert emission_field.height == height, f"Expected height: {height}. Got: {emission_field.height}"


class CropTransform(EmissionFieldTransform):
    def __init__(self, center_offset_x: int, center_offset_y: int, width: int, height: int) -> None:
        self._center_offset_x = center_offset_x
        self._center_offset_y = center_offset_y
        self._width = width
        self._height = height

    def __call__(self, emission_field: CityEmissionField) -> CityEmissionField:
        center_x = emission_field.width // 2 + self._center_offset_x
        center_y = emission_field.height // 2 + self._center_offset_y

        start_x = center_x - self._width // 2
        start_y = center_y - self._height // 2

        _crop_emission_field(
            emission_field=emission_field,
            start_x=start_x,
            start_y=start_y,
            width=self._width,
            height=self._height,
        )

        return emission_field


class CenterCropTransform(EmissionFieldTransform):
    def __init__(self, width: int, height: int) -> None:
        self._crop_transform = CropTransform(center_offset_x=0, center_offset_y=0, width=width, height=height)

    def __call__(self, emission_field: CityEmissionField) -> CityEmissionField:
        return self._crop_transform(emission_field)


class RandomCropTransform(EmissionFieldTransform):
    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height

    def __call__(self, emission_field: CityEmissionField) -> CityEmissionField:
        max_x = emission_field.width - self._width
        max_y = emission_field.height - self._height

        start_x = random.randint(0, max_x)
        start_y = random.randint(0, max_y)

        _crop_emission_field(
            emission_field=emission_field,
            start_x=start_x,
            start_y=start_y,
            width=self._width,
            height=self._height,
        )

        return emission_field
