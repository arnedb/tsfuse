import pytest
import numpy as np

from tsfuse.data.synthetic import dimensions, brownian
from tsfuse.data import Collection
from tsfuse.transformers.geometry import *


@pytest.fixture
def x():
    return brownian()


def test_norm_1(x):
    result = Norm(p=1).transform(x)
    for i, a in dimensions(x):
        actual = result.values[i]
        expected = np.sum(np.abs(a))
        np.testing.assert_almost_equal(actual, expected)


def test_norm_2(x):
    result = Norm(p=2).transform(x)
    for i, a in dimensions(x):
        actual = result.values[i]
        expected = np.sqrt(np.sum(np.square(a)))
        np.testing.assert_almost_equal(actual, expected)


def test_resultant(x):
    result = Resultant().transform(x)
    for i, a in dimensions(x):
        actual = result.values[i]
        expected = np.sqrt(np.sum(np.square(a)))
        np.testing.assert_almost_equal(actual, expected)


def test_angle_half():
    p1 = Collection([1, 0])
    p2 = Collection([0, 0])
    p3 = Collection([-2, 0])
    actual = Angle().transform(p1, p2, p3, ignore_preconditions=True).values
    np.testing.assert_almost_equal(np.mod(actual, np.pi), 0)


def test_angle_perpendicular():
    p1 = Collection([1, 0])
    p2 = Collection([0, 0])
    p3 = Collection([0, 2])
    actual = Angle().transform(p1, p2, p3, ignore_preconditions=True).values
    np.testing.assert_almost_equal(np.mod(actual, np.pi), np.pi / 2)
