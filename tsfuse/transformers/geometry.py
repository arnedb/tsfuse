import numpy as np

from tsfuse.data.units import units
from ..computation import Transformer, Graph
from .mathematics import Sum, ArcCos
from .util import apply_to_axis

__all__ = [
    'Norm',
    'Resultant',
    'Angle',
]


class Norm(Transformer):
    def __init__(self, *parents, p=2, axis='dimensions', **kwargs):
        super(Norm, self).__init__(*parents, **kwargs)
        self.p = p
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def apply(self, x):
        def calculator(a):
            return np.linalg.norm(a, ord=self.p, axis=-1, keepdims=True)

        return apply_to_axis(calculator, x, axis=self.axis)


class Resultant(Transformer):
    def __init__(self, *parents, axis='dimensions', **kwargs):
        super(Resultant, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
            lambda x: x.shape[2] > 1,
        ]

    def graph(self, x):
        return Graph(Norm(x, p=2, axis=self.axis))


class Angle(Transformer):
    def __init__(self, *parents, axis='dimensions', **kwargs):
        super(Angle, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 3,
            lambda *collections: all(c.shape[2] > 1 for c in collections),
            lambda p1, p2, p3: np.issubdtype(p1.dtype, np.float64),
            lambda p1, p2, p3: np.issubdtype(p2.dtype, np.float64),
            lambda p1, p2, p3: np.issubdtype(p3.dtype, np.float64),
            lambda p1, p2, p3: len(set(p.shape for p in (p1, p2, p3))) == 1,
            lambda p1, p2, p3: p1.tags['quantity'] == 'position',
            lambda p1, p2, p3: p2.tags['quantity'] == 'position',
            lambda p1, p2, p3: p3.tags['quantity'] == 'position',
        ]

    def graph(self, p1, p2, p3):
        v1 = p2 - p1
        l1 = Norm(v1, axis=self.axis)
        v2 = p3 - p2
        l2 = Norm(v2, axis=self.axis)
        dot = Sum(v1 * v2, axis=self.axis)
        theta = ArcCos(dot / (l1 * l2))
        return Graph(theta)
