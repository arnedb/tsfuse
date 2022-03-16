import numpy as np
from scipy import signal

from ..computation import Transformer
from .util import apply_to_axis

__all__ = [
    'NumberPeaks',
    'NumberPeaksCWT',
]

"""
Add later:
- DetectPeaks
- DetectPeaksCWT
- PeakDistanceMean
- PeakDistanceMeanCWT
- PeakDistanceVariance
- PeakDistanceVarianceCWT
"""


class NumberPeaks(Transformer):
    """
    Number of peaks

    Parameters
    ----------
    support : int, optional
        Minimum support of each peak. Default: 1
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """
    def __init__(self, *parents, support=1, axis=None, **kwargs):
        super(NumberPeaks, self).__init__(*parents, **kwargs)
        self.support = support
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def transform(self, x, **kwargs):
        """
        For each series in **x**, compute the number of peaks that have a
        support larger than the  given minimum support.
        The support of a peak is defined as the length of the largest
        subsequence around the peak where the peak has the largest value.
        """
        return super().transform(x, **kwargs)

    def apply(self, x):
        def calculator(a):
            n = self.support
            l = a.shape[-1] - n * 2
            is_peak = np.ones((a.shape[0], a.shape[1], a.shape[2] - n * 2), dtype=bool)
            for i in range(1, n + 1):
                is_peak &= a[:, :, n - i:][:, :, :l] < a[:, :, n:-n]
                is_peak &= a[:, :, n + i:][:, :, :l] < a[:, :, n:-n]
            return np.sum(is_peak, axis=-1, keepdims=True)

        return apply_to_axis(calculator, x, axis=self.axis)


class NumberPeaksCWT(Transformer):
    """
    Number of peaks estimated using a continous wavelet transform

    Parameters
    ----------
    max_width : int, optional
        Maximum width of the wavelet. Default: 1
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """
    def __init__(self, *parents, max_width=1, axis=None, **kwargs):
        super(NumberPeaksCWT, self).__init__(*parents, **kwargs)
        self.max_width = max_width
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def transform(self, x, **kwargs):
        """
        For each series in **x**, estimate the number of peaks using
        ``scipy.signal.find_peaks_cwt`` where ``widths = [1, ..., max_width]``
        """
        return super().transform(x, **kwargs)

    def apply(self, x):
        def calculator1d(a):
            nnan = ~np.isnan(a)
            result = np.full(a.shape, fill_value=np.nan)
            a = a[nnan]
            widths = np.arange(1, self.max_width + 1)
            peaks = signal.find_peaks_cwt(a, widths, wavelet=signal.ricker)
            result[nnan] = np.array([len(peaks)])
            return result

        def calculator(a):
            result = np.apply_along_axis(calculator1d, -1, a)
            return result

        return apply_to_axis(calculator, x, axis=self.axis)
