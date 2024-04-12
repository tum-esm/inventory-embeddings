from embeddings.dataset.city_emission_field import CityEmissionField


class CropTransform:
    def __init__(self, center_offset_x: int, center_offset_y: int, width: int, height: int) -> None:
        self._center_offset_x = center_offset_x
        self._center_offset_y = center_offset_y
        self._width = width
        self._height = height

    def __call__(self, emission_field: CityEmissionField) -> CityEmissionField:
        emission_field.crop(
            center_offset_x=self._center_offset_x,
            center_offset_y=self._center_offset_y,
            width=self._width,
            height=self._height,
        )
        return emission_field
