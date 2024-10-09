from typing import Any

_LEARNING_RATES_256 = {
    10: 1.5e-4,
    25: 5e-4,
    50: 1e-3,
    100: 1.8e-3,
    250: 2.5e-3,
    500: 4e-3,
    1_000: 5.5e-3,
    2_500: 7e-3,
    5_000: 1.5e-2,
    10_000: 3e-2,
    12_500: 4e-2,
}

_LEARNING_RATES_512 = {
    500: 2e-2,
    1_000: 2e-2,
    2_500: 5e-2,
    5_000: 7.5e-2,
    10_000: 7.5e-2,
    12_500: 7.5e-2,
}

_LEARNING_RATES_1024 = {
    500: 1e-3,
    1_000: 6e-2,
    2_500: 1e-1,
    5_000: 1e-1,
    10_000: 2e-1,
    12_500: 2e-1,
}

_LEARNING_RATES_2048 = {
    500: 5e-4,
    1_000: 6e-2,
    2_500: 1e-1,
    5_000: 1e-1,
    10_000: 2e-1,
    12_500: 2e-1,
}

LEARNING_RATES: dict[int, dict[int, Any]] = {
    256: _LEARNING_RATES_256,
    512: _LEARNING_RATES_512,
    1024: _LEARNING_RATES_1024,
    2048: _LEARNING_RATES_2048,
}
