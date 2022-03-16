import numpy as np

from ..computation import Transformer
from ..computation.nodes import Greater, Less
from ..data import Collection

__all__ = [
    'Greater',
    'Less',
    'Equal',
    'NotEqual',
]


class Equal(Transformer):
    """
    Element-wise equality comparison
    """

    def __init__(self, *parents, **kwargs):
        super(Equal, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 2,
        ]

    def transform(self, x, y, **kwargs):
        """
        Compute :math:`x = y`
        """
        return super().transform(x, y, **kwargs)

    @staticmethod
    def apply(x, y):
        x, y = _collections(x, y)
        values = np.array(x.values == y.values, dtype=bool)
        return _result(x, y, values)


class NotEqual(Transformer):
    """
    Element-wise inequality comparison
    """

    def __init__(self, *parents, **kwargs):
        super(NotEqual, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 2,
        ]

    def transform(self, x, y, **kwargs):
        """
        Compute :math:`x \\neq y`
        """
        return super().transform(x, y, **kwargs)

    @staticmethod
    def apply(x, y):
        x, y = _collections(x, y)
        values = np.array(x.values != y.values, dtype=bool)
        return _result(x, y, values)


def _collections(x, y):
    if not isinstance(x, Collection):
        x = Collection.from_array(np.array([[[[x]]]]))
    if not isinstance(y, Collection):
        y = Collection.from_array(np.array([[[[y]]]]))
    return x, y


def _result(x, y, values):
    if values.shape == x.shape:
        return Collection.from_array(values, time=x.time, dims=x.dims)
    else:
        return Collection.from_array(values, time=y.time, dims=y.dims)
