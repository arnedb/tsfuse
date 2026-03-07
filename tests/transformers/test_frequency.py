import warnings

import numpy as np
import pytest
import scipy.signal as signal

from tsfuse.data.synthetic import brownian, series
from tsfuse.transformers.frequency import *
from tsfuse.transformers.statistics import *


@pytest.fixture
def x():
    return brownian()


def test_fft_real(x):
    result = FFT(attr='real').transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.fft.rfft(a).real
        np.testing.assert_almost_equal(actual, expected)


def test_fft_imag(x):
    result = FFT(attr='imag').transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.fft.rfft(a).imag
        np.testing.assert_almost_equal(actual, expected)


def test_fft_abs(x):
    result = FFT(attr='abs').transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.abs(np.fft.rfft(a))
        np.testing.assert_almost_equal(actual, expected)


def test_fft_angle(x):
    result = FFT(attr='angle').transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.angle(np.fft.rfft(a), deg=True)
        np.testing.assert_almost_equal(actual, expected)


def test_fft_moments(x):
    def moment(a, moment):
        return a.dot(np.arange(len(a)) ** moment) / np.sum(a)

    def mean(a):
        return moment(a, 1)

    def variance(a):
        return moment(a, 2) - mean(a) ** 2

    def skewness(a):
        return (moment(a, 3) - 3 * mean(a) * variance(a) - mean(a) ** 3) / variance(a) ** (1.5)

    def kurtosis(a):
        return ((moment(a, 4) - 4 * mean(a) * moment(a, 3)
                 + 6 * moment(a, 2) * mean(a) ** 2 - 3 * mean(a))
                / variance(a) ** 2)

    fft = FFT().transform(x)
    result_mean = SpectralMean().transform(x)
    result_variance = SpectralVariance().transform(x)
    result_skewness = SpectralSkewness().transform(x)
    result_kurtosis = SpectralKurtosis().transform(x)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, a in series(x):
            np.testing.assert_allclose(result_mean.values[i], mean(a))
            np.testing.assert_allclose(result_variance.values[i], variance(a))
            np.testing.assert_allclose(result_skewness.values[i], skewness(a))
            np.testing.assert_allclose(result_kurtosis.values[i], kurtosis(a))


def _ricker(points, a):
    A = 2 / (np.sqrt(3 * a) * (np.pi**0.25))
    wsq = a**2
    vec = np.arange(0, points) - (points - 1.0) / 2
    tsq = vec**2
    mod = 1 - tsq / wsq
    gauss = np.exp(-tsq / (2 * wsq))
    return A * mod * gauss


def test_cwt_ricker_width_1(x):
    result = CWT(wavelet='ricker', width=1).transform(x)
    for i, a in series(x):
        actual = result.values[i]
        wavelet = _ricker(min(10, len(a)), 1)
        expected = signal.convolve(a, wavelet, mode='same')
        np.testing.assert_almost_equal(actual, expected)


def test_cwt_ricker_width_2(x):
    result = CWT(wavelet='ricker', width=2).transform(x)
    for i, a in series(x):
        actual = result.values[i]
        wavelet = _ricker(min(20, len(a)), 2)
        expected = signal.convolve(a, wavelet, mode='same')
        np.testing.assert_almost_equal(actual, expected)


def test_power_spectral_density(x):
    result = PowerSpectralDensity().transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = signal.welch(a, nperseg=min(len(a), 256))[1]
        np.testing.assert_almost_equal(actual, expected)
