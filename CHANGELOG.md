# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-03-07

### Changed

- **BREAKING:** Minimum Python version is now 3.10.
- Consolidated all packaging configuration into `pyproject.toml` (PEP 621).
- Removed `setup.py`, `setup.cfg`, and `MANIFEST.in`.
- Removed `six` dependency — no longer needed for Python 2 compatibility.
- Modernized all class definitions (removed `object` base class, use `abc.ABC`).
- Modernized all `super()` calls (no-argument form).
- Updated all dependency lower bounds to recent, secure versions.
- Added `py.typed` marker for PEP 561 typed package support.
- Added type annotations to public APIs.
- Added `ruff` for linting and formatting.
- Added `mypy` configuration for gradual type checking.
- Added `pre-commit` configuration.
- Modernized GitHub Actions CI workflow (actions v4, matrix testing, linting job).
- Added `CHANGELOG.md`.
- Improved `README.md` with additional badges and contributing section.

### Removed

- `six` dependency.
- `setup.py`, `setup.cfg`, `MANIFEST.in`.
- Python 2 compatibility patterns (`@six.add_metaclass`, `super(Class, self)`).

## [0.1.2] - 2022-01-01

### Added

- Initial public release.
- Feature construction from multiple time series.
- Computation graph for reproducible feature pipelines.

[0.2.0]: https://github.com/arnedb/tsfuse/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/arnedb/tsfuse/releases/tag/v0.1.2
