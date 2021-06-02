import pytest
import numpy as np

from tsfuse.data.synthetic import series, brownian
from tsfuse.data import Collection
from tsfuse.transformers.queries import *


@pytest.fixture
def x():
    return brownian()


def test_slice_first(x):
    result = Slice(i=0).transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = a[0]
        np.testing.assert_equal(actual, expected)


def test_slice_second(x):
    result = Slice(i=1).transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = a[1]
        np.testing.assert_equal(actual, expected)


def test_slice_last(x):
    result = Slice(i=-1).transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = a[-1]
        np.testing.assert_equal(actual, expected)


def test_aggregate_mean():
    x = Collection.from_array([1, 2, 3, 3, 5])
    actual = Aggregate(size=2, agg='mean', axis='dims').transform(x).values
    np.testing.assert_equal(actual[0, 0, :], [1.5, 3, 5])


def test_aggregate_min():
    x = Collection.from_array([1, 2, 3, 3, 5])
    actual = Aggregate(size=2, agg='min', axis='dims').transform(x).values
    np.testing.assert_equal(actual[0, 0, :], [1, 3, 5])


def test_aggregate_max():
    x = Collection.from_array([1, 2, 3, 3, 5])
    actual = Aggregate(size=2, agg='max', axis='dims').transform(x).values
    np.testing.assert_equal(actual[0, 0, :], [2, 3, 5])
