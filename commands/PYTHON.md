# Python Related Commands

Collection of useful Python development commands.

## Environment and Package Management

### Conda for Environments

```bash
# Create conda environment
conda create -n myenv python=3.10

# Activate conda environment
conda activate myenv

# Deactivate conda environment
conda deactivate

# List all conda environments
conda env list

# Export environment to YAML
conda env export > environment.yml

# Create environment from YAML
conda env create -f environment.yml
```

### Pip for Packages

```bash
# Install packages
pip install package_name
pip install -r requirements.txt

# Install specific version
pip install package_name==1.2.3

# Update pip
python -m pip install --upgrade pip

# Show installed packages
pip list
pip freeze > requirements.txt

# Show version metadata
pip show package_name
```

### Project Setup with setuptools

```bash
# Create pyproject.toml

```toml
[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my_package"
version = "0.1.0"
description = "My Python package"
authors = [{name = "Your Name", email = "your.email@example.com"}]
requires-python = ">=3.8"
dependencies = [
    "requests",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "black",
    "isort",
]

[tool.black]
line-length = 88
target-version = ["py38"]

[tool.isort]
profile = "black"
line_length = 88
```

## Build package

```bash
python -m build
```

or

```bash
pip install -e .
```

## Code Formatting and Linting

```bash
# Format code with black
black .

# Sort imports with isort
isort .
```

## Testing with pytest

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_specific.py::test_function

# Run with coverage and generate coverage report
pytest --cov=my_package --cov-report=html

# Run tests that match a keyword
pytest -k "keyword"

# Run tests and stop on first failure
pytest -x

# Run tests and show local variables on failure
pytest --showlocals
```
