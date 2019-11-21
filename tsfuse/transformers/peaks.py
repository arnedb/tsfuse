import numpy as np
from scipy import signal

from ..computation import Transformer
from .util import apply_to_axis

__all__ = [
    'NumberPeaks',
    'NumberPeaksCWT',
]

"""
Add:
- DetectPeaks
- DetectPeaksCWT
- PeakDistanceMean
- PeakDistanceMeanCWT
- PeakDistanceVariance
- PeakDistanceVarianceCWT
"""


class NumberPeaks(Transformer):
    def __init__(self, *parents, support=1, axis=None, **kwargs):
        super(NumberPeaks, self).__init__(*parents, **kwargs)
        self.support = support
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def apply(self, x):
        def calculator(a):
            n = self.support
            l = a.shape[-1] - n * 2
            is_peak = np.ones((a.shape[0], a.shape[1], a.shape[2] - n * 2), dtype=bool)
            for i in range(1, n + 1):
                is_peak &= a[:, :, n - i:][:, :, :l] < a[:, :, n:-n]
                is_peak &= a[:, :, n + i:][:, :, :l] < a[:, :, n:-n]
            return np.sum(is_peak, keepdims=True)

        return apply_to_axis(calculator, x, axis=self.axis)


class NumberPeaksCWT(Transformer):
    def __init__(self, *parents, max_width=1, axis=None, **kwargs):
        super(NumberPeaksCWT, self).__init__(*parents, **kwargs)
        self.max_width = max_width
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

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
