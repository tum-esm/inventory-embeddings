from abc import ABC, abstractmethod

from embeddings.dataset.city_emission_field import CityEmissionField


class EmissionFieldTransform(ABC):
    @abstractmethod
    def __call__(self, emission_field: CityEmissionField) -> CityEmissionField: ...
