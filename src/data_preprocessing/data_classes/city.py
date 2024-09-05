from dataclasses import dataclass


@dataclass
class City:
    name: str
    lat: float
    lon: float
    population: int
