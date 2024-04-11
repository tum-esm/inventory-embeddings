# Installation

### Installation With Default PyTorch Configuration
```bash
poetry install
```

### Alternative Installation of PyTorch
Setup dependencies without torch
```bash
poetry install --without torch
```

Get the installation command from the [official website](https://pytorch.org/get-started/locally/).

Example:
```bash
poetry run pip install torch --index-url https://download.pytorch.org/whl/cu121
```

# Usage
### Preprocessing of TNO Data
```bash
poetry run preprocess-tno
```

# Development

### Installation of pre-commit hooks
```bash
poetry run pre-commit install
```


