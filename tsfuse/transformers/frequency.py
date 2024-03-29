import numpy as np
import scipy.signal as signal

from ..computation import Transformer
from .util import apply_to_axis

__all__ = [
    'FFT',
    'CWT',
    'PowerSpectralDensity',
]


class FFT(Transformer):
    """
    Fast Fourier transform

    Parameters
    ----------
    attr : {'real', 'imag', 'abs', 'angle'}, optional
        Return the real part ('real'), imginary part ('imag'),
        absolute value ('abs'), or angle in degrees ('angle').
        Default: 'abs'
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: 'time'
    """
    def __init__(self, *parents, attr='abs', axis='time', **kwargs):
        super(FFT, self).__init__(*parents, **kwargs)
        self.attr = attr
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def transform(self, x, **kwargs):
        """
        Compute the fast Fourier transform of each time series in **x**

        Parameters
        ----------
        x : Collection
            Time series data.
        """
        return super().transform(x, **kwargs)

    def apply(self, x):
        def calculator1d(a):
            nnan = ~np.isnan(a)
            a = a[nnan]
            return np.fft.rfft(a)

        def calculator(a):
            fft = np.apply_along_axis(calculator1d, -1, a)
            if self.attr == 'real':
                return fft.real
            elif self.attr == 'imag':
                return fft.imag
            elif self.attr == 'abs':
                return np.abs(fft)
            elif self.attr == 'angle':
                return np.angle(fft, deg=True)

        return apply_to_axis(calculator, x, axis=self.axis)


class CWT(Transformer):
    """
    Continuous wavelet transform

    Parameters
    ----------
    wavelet : {'ricker'}, optional
        Wavelet type. Default: 'ricker'
    width : int, optional
        Wavelet width. Default: 1
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: 'time'
    """
    def __init__(self, *parents, wavelet='ricker', width=1, axis=None, **kwargs):
        super(CWT, self).__init__(*parents, **kwargs)
        self.wavelet = wavelet
        self.width = width
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def transform(self, x, **kwargs):
        """
        Compute continous wavelet transform for each time series in **x**

        Parameters
        ----------
        x : Collection
            Time series data.
        """
        return super().transform(x, **kwargs)

    def apply(self, x):
        def calculator1d(a):
            nnan = ~np.isnan(a)
            a = a[nnan]
            if self.wavelet == 'ricker':
                wavelet = signal.ricker(min(10 * self.width, len(a)), self.width)
            else:
                raise NotImplementedError()
            return signal.convolve(a, wavelet, mode='same')

        def calculator(a):
            return np.apply_along_axis(calculator1d, -1, a)

        return apply_to_axis(calculator, x, axis=self.axis)


class PowerSpectralDensity(Transformer):
    """
    Power spectral density

    Parameters
    ----------
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: 'time'
    """
    def __init__(self, *parents, axis=None, **kwargs):
        super(PowerSpectralDensity, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def transform(self, x, **kwargs):
        """
        Compute power spectral density using Welch's method,
        for each time series in **x**

        Parameters
        ----------
        x : Collection
            Time series data.
        """
        return super().transform(x, **kwargs)

    def apply(self, x):
        def calculator1d(a):
            nnan = ~np.isnan(a)
            return signal.welch(a[nnan], nperseg=min(a.shape[0], 256))[1]

        def calculator(a):
            return np.apply_along_axis(calculator1d, -1, a)

        return apply_to_axis(calculator, x, axis=self.axis)
