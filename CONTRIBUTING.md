# Contributing to TSFuse

Thank you for your interest in contributing to TSFuse! This document provides
guidelines and instructions for contributing.

## Development Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/arnedb/tsfuse.git
   cd tsfuse
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install in development mode:**

   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks:**

   ```bash
   pre-commit install
   ```

## Code Quality

This project uses the following tools to maintain code quality:

- **[Ruff](https://docs.astral.sh/ruff/)** for linting and formatting.
- **[Mypy](https://mypy.readthedocs.io/)** for static type checking.
- **[Pytest](https://docs.pytest.org/)** for testing.

### Running Checks Locally

```bash
# Lint and format
ruff check .
ruff format .

# Type check
mypy tsfuse

# Run tests
pytest

# Run tests with coverage
pytest --cov=tsfuse --cov-report=term-missing
```

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Make your changes, ensuring all tests pass.
3. Add tests for any new functionality.
4. Update documentation as needed.
5. Submit a pull request with a clear description of the changes.

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new transformer for wavelet decomposition
fix: handle NaN values in AutoCorrelation
docs: update quickstart example
test: add edge case tests for Graph.transform
```

## Reporting Issues

When reporting bugs, please include:

- Python version and OS.
- Minimal reproducible example.
- Expected vs. actual behavior.
- Full traceback if applicable.
