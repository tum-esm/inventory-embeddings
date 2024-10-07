from typing import Any

_MUNICH_DATA = {
    "2048": {
        5: 2.5e-4,
        10: 2.5e-4,
        15: 3e-4,
        20: 3e-3,
        25: 5e-3,
        30: 1e-2,
        35: 2e-2,
        40: 0.1,
    },
    "2048_munich": {
        5: 5e-4,
        10: 8e-4,
        15: 1e-3,
        20: 5e-3,
        25: 1e-2,
        30: 0.1,
        35: 0.1,
        40: 0.1,
    },
}

LAMBDA_PER_SNR: dict[str, dict[str, dict[int, Any]]] = {
    "Munich": _MUNICH_DATA,
}
