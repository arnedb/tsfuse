import numpy as np
import scipy.signal as signal

from ..computation import Transformer
from .util import apply_to_axis

__all__ = [
    'Resample',
]


class Resample(Transformer):
    """
    Resample

    Parameters
    ----------
    num : int
        New number of samples.
    axis : {'time', 'dims'}, optional
        Time direction: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """
    def __init__(self, *parents, num=None, axis=None, **kwargs):
        super(Resample, self).__init__(*parents, **kwargs)
        self.num = num
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
        ]

    def transform(self, x, **kwargs):
        """
        Resample **x** in the time direction to **num** samples
        using ``scipy.signal.resample``
        """
        return super().transform(x, **kwargs)

    def apply(self, x):
        def calculator(a):
            return signal.resample(a, num=self.num, axis=-1)

        return apply_to_axis(calculator, x, axis=self.axis)
