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
    Element-wise equality comparison.

    Preconditions:

    - Number of inputs: 1
    """

    def __init__(self, *parents, **kwargs):
        super(Equal, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 2,
        ]

    @staticmethod
    def apply(x, y):
        x, y = _collections(x, y)
        values = np.array(x.values == y.values, dtype=bool)
        return _result(x, y, values)


class NotEqual(Transformer):
    """
    Element-wise inequality comparison.

    Preconditions:

    - Number of inputs: 1
    """

    def __init__(self, *parents, **kwargs):
        super(NotEqual, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 2,
        ]

    @staticmethod
    def apply(x, y):
        x, y = _collections(x, y)
        values = np.array(x.values != y.values, dtype=bool)
        return _result(x, y, values)


def _collections(x, y):
    if not isinstance(x, Collection):
        x = Collection(np.array([[[[x]]]]))
    if not isinstance(y, Collection):
        y = Collection(np.array([[[[y]]]]))
    return x, y


def _result(x, y, values):
    if values.shape == x.shape:
        return Collection(values, index=x.index, dimensions=x.dimensions)
    else:
        return Collection(values, index=y.index, dimensions=y.dimensions)
