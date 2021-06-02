import pytest
import numpy as np
from scipy import signal

from tsfuse.data.synthetic import series, brownian
from tsfuse.transformers.peaks import *
from tsfuse.data import Collection


@pytest.fixture
def x():
    return brownian()


def test_number_of_peaks_support_1():
    x = Collection.from_array([1, 2, 1, 2, 3, 2, 3])
    actual = NumberPeaks(support=1).transform(x).values
    np.testing.assert_equal(actual, 2)


def test_number_of_peaks_support_1_zero():
    x = Collection.from_array([1, 1, 1, 1])
    actual = NumberPeaks(support=1).transform(x).values
    np.testing.assert_equal(actual, 0)


def test_number_of_peaks_support_2():
    x = Collection.from_array([1, 2, 3, 2, 1, 0, 1, 0])
    actual = NumberPeaks(support=2).transform(x).values
    np.testing.assert_equal(actual, 1)


def test_number_of_peaks_support_2_zero():
    x = Collection.from_array([3, 2, 1, 0])
    actual = NumberPeaks(support=2).transform(x).values
    np.testing.assert_equal(actual, 0)


def test_number_of_peaks_cwt(x):
    result = NumberPeaksCWT(max_width=5).transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = len(signal.find_peaks_cwt(a, np.arange(1, 6), wavelet=signal.ricker))
        np.testing.assert_almost_equal(actual, expected)
