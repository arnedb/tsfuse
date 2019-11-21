import pytest
import numpy as np
from scipy import signal

from tsfuse.data.synthetic import series, brownian
from tsfuse.transformers.sampling import *


@pytest.fixture
def x():
    return brownian()


def test_resample_up(x):
    num = x.shape[1] * 2
    result = Resample(num=num).transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = signal.resample(a, num=num)
        np.testing.assert_almost_equal(actual, expected)


def test_resample_down(x):
    num = int(x.shape[1] / 2)
    result = Resample(num=num).transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = signal.resample(a, num=num)
        np.testing.assert_almost_equal(actual, expected)
