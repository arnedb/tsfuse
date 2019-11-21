import numpy as np
import scipy.signal as signal

from ..computation import Transformer
from .util import apply_to_axis

__all__ = [
    'Resample',
]


class Resample(Transformer):
    def __init__(self, *parents, num=None, axis=None, **kwargs):
        super(Resample, self).__init__(*parents, **kwargs)
        self.num = num
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
        ]

    def apply(self, x):
        def calculator(a):
            return signal.resample(a, num=self.num, axis=-1)

        return apply_to_axis(calculator, x, axis=self.axis)
