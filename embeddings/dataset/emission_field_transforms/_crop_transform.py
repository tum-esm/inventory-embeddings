from embeddings.dataset.city_emission_field import CityEmissionField

from ._emission_field_transform import EmissionFieldTransform


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
        end_x = start_x + self._width
        end_y = start_y + self._height

        if any([start_x < 0, start_y < 0, end_x > emission_field.width, end_y > emission_field.height]):
            value_error = "Tried to crop beyond field border!"
            raise ValueError(value_error)

        emission_field.co2_ff_field = emission_field.co2_ff_field[:, start_x:end_x, start_y:end_y]
        emission_field.lat_lon_array = emission_field.lat_lon_array[start_x:end_x, start_y:end_y, :]

        assert emission_field.width == self._width
        assert emission_field.height == self._height

        return emission_field
