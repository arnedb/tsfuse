import pytest
import numpy as np

from tsfuse.data.synthetic import series, brownian
from tsfuse.data import Collection
from tsfuse.transformers.mathematics import *


@pytest.fixture
def x():
    return brownian()


@pytest.fixture
def y():
    return brownian()


def test_square(x):
    result = Square().transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.square(a)
        np.testing.assert_almost_equal(actual, expected)


def test_exponent(x):
    result = Exponent(a=3).transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.power(a, 3)
        np.testing.assert_almost_equal(actual, expected)


def test_abs(x):
    result = Abs().transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.abs(a)
        np.testing.assert_almost_equal(actual, expected)


def test_cum_sum(x):
    result = CumSum().transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.cumsum(a)
        np.testing.assert_almost_equal(actual, expected)


def test_diff(x):
    result = Diff().transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.diff(a)
        np.testing.assert_almost_equal(actual, expected)


def test_roots():
    x = Collection.from_array([-1, 0, 1])
    actual = Roots().transform(x).values
    np.testing.assert_almost_equal(actual.flatten(), [1, -1])


def test_average(x, y):
    result = Average().transform(x, y)
    for i, a, b in series(x, y):
        actual = result.values[i]
        expected = (a + b) / 2
        np.testing.assert_almost_equal(actual, expected)


def test_difference_absolute(x, y):
    result = Difference().transform(x, y)
    for i, a, b in series(x, y):
        actual = result.values[i]
        expected = np.abs(b - a)
        np.testing.assert_almost_equal(actual, expected)


def test_difference_relative(x, y):
    result = Difference(rel=True).transform(x, y)
    for i, a, b in series(x, y):
        actual = result.values[i]
        with np.errstate(divide='ignore', invalid='ignore'):
            expected = np.abs((b - a) / a)
        np.testing.assert_almost_equal(actual, expected)


def test_ratio_absolute(x, y):
    result = Ratio().transform(x, y)
    for i, a, b in series(x, y):
        actual = result.values[i]
        with np.errstate(divide='ignore', invalid='ignore'):
            expected = a / b
        np.testing.assert_almost_equal(actual, expected)
