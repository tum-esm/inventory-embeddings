# Install
```bash
poetry install
```

# Install pre-commit hooks
```bash
poetry run pre-commit install
```

# Alternative Installation of PyTorch
Setup dependencies without torch
```bash
poetry install --without torch
```

Get the installation command from the [official website](https://pytorch.org/get-started/locally/).

Example:
```bash
poetry run pip install torch --index-url https://download.pytorch.org/whl/cu121
```
