import numpy as np

from ..computation import Transformer
from ..data import Collection
from .util import transform_axis, apply_to_axis
from .calculators.queries import *

__all__ = [
    'Count',
    'Slice',
    'Aggregate',
]


class Count(Transformer):
    def __init__(self, *parents, axis=None, **kwargs):
        super(Count, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.bool_),
        ]

    def apply(self, x):
        def calculator(a):
            return np.nansum(a, keepdims=True, axis=-1)

        return apply_to_axis(calculator, x, axis=self.axis)


class Slice(Transformer):
    def __init__(self, *parents, i=None, axis=None, **kwargs):
        super(Slice, self).__init__(*parents, **kwargs)
        self.i = i
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
        ]

    def apply(self, x):
        a = transform_axis(x, axis=self.axis)
        if isinstance(self.i, int):
            if self.i != -1:
                s = slice(self.i, self.i + 1)
            else:
                s = slice(self.i, None)
        elif isinstance(self.i, tuple):
            s = slice(*self.i)
        else:
            s = self.i
        if a == 0:
            values = x.values[s, :, :]
            index = x.index[s, :]
            dimensions = x.dimensions[:]
        elif a == 1:
            values = x.values[:, s, :]
            index = x.index[:, s]
            dimensions = x.dimensions[:]
        else:
            values = x.values[:, :, s]
            index = x.index[:, :]
            dimensions = x.dimensions[s]
        if values.shape[a] == 0:
            return None
        return Collection(values, index=index, dimensions=dimensions)


class Aggregate(Transformer):
    def __init__(self, *parents, size=10, agg='mean', axis='timestamps', **kwargs):
        super(Aggregate, self).__init__(*parents, **kwargs)
        self.size = size
        self.agg = agg
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def apply(self, x):
        def calculator(a):
            if self.agg == 'mean':
                return aggregate(a, size=self.size, agg=0)
            elif self.agg == 'var':
                return aggregate(a, size=self.size, agg=1)
            elif self.agg == 'min':
                return aggregate(a, size=self.size, agg=2)
            elif self.agg == 'max':
                return aggregate(a, size=self.size, agg=3)

        return apply_to_axis(calculator, x, axis=self.axis)