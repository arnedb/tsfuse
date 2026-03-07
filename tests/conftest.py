"""Shared test fixtures for TSFuse test suite."""

from __future__ import annotations

import numpy as np
import pytest

from tsfuse.data import Collection
from tsfuse.data.synthetic import brownian


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_collection() -> Collection:
    """A small deterministic Collection for unit tests."""
    return brownian(N=5, t=20, d=2, random_state=42)


@pytest.fixture
def univariate_collection() -> Collection:
    """A univariate (d=1) Collection."""
    return brownian(N=5, t=20, d=1, random_state=123)
