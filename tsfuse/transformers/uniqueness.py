import numpy as np

from ..computation import Graph, Transformer, Constant
from .statistics import ArgMin, ArgMax
from .util import apply_to_axis

__all__ = [
    'HasDuplicate',
    'HasDuplicateMin',
    'HasDuplicateMax',
    'NumberUniqueValues',
    'SumReoccurringDataPoints',
    'SumReoccurringValues',
]


class NumberUniqueValues(Transformer):

    def __init__(self, *parents, rel=True, axis=None, **kwargs):
        super(NumberUniqueValues, self).__init__(*parents, **kwargs)
        self.rel = rel
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
        ]

    def apply(self, x):
        def calculator1d(a):
            nnan = ~np.isnan(a)
            result = np.full(a.shape, fill_value=np.nan)
            a = a[nnan]
            n = a[~np.isnan(a)].size
            if n > 0:
                unique = np.unique(a).size
                if self.rel:
                    unique = unique / n
                result[nnan] = np.array([unique])
                return result
            else:
                result[nnan] = np.array([np.nan])
                return result

        def calculator(a):
            return np.apply_along_axis(calculator1d, -1, a)

        return apply_to_axis(calculator, x, axis=self.axis)


class HasDuplicate(Transformer):

    def __init__(self, *parents, axis=None, **kwargs):
        super(HasDuplicate, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
        ]

    def graph(self, x):
        return Graph(NumberUniqueValues(x, rel=True, axis=self.axis) < Constant(1))


class HasDuplicateMin(Transformer):

    def __init__(self, *parents, axis=None, **kwargs):
        super(HasDuplicateMin, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
        ]

    def graph(self, x):
        return Graph(
            ArgMin(x, first=True, axis=self.axis) < ArgMin(x, first=False, axis=self.axis)
        )


class HasDuplicateMax(Transformer):

    def __init__(self, *parents, axis=None, **kwargs):
        super(HasDuplicateMax, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
        ]

    def graph(self, x):
        return Graph(
            ArgMax(x, first=True, axis=self.axis) < ArgMax(x, first=False, axis=self.axis)
        )


class SumReoccurringValues(Transformer):

    def __init__(self, *parents, axis=None, **kwargs):
        super(SumReoccurringValues, self).__init__(*parents, **kwargs)
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
            unique, counts = np.unique(a, return_counts=True)
            counts[counts < 2] = 0
            counts[counts > 1] = 1
            result[nnan] = np.sum(counts * unique, keepdims=True)
            return result

        def calculator(a):
            return np.apply_along_axis(calculator1d, -1, a)

        return apply_to_axis(calculator, x, axis=self.axis)


class SumReoccurringDataPoints(Transformer):

    def __init__(self, *parents, axis=None, **kwargs):
        super(SumReoccurringDataPoints, self).__init__(*parents, **kwargs)
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
            unique, counts = np.unique(a, return_counts=True)
            counts[counts < 2] = 0
            result[nnan] = np.sum(counts * unique, keepdims=True)
            return result

        def calculator(a):
            return np.apply_along_axis(calculator1d, -1, a)

        return apply_to_axis(calculator, x, axis=self.axis)
