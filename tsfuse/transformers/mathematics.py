import warnings
import numpy as np

from ..data import units
from ..computation import Transformer, Graph
from .util import apply, apply_to_axis
from ..computation.nodes import Add, Subtract, Multiply, Divide, Constant
from .queries import Slice

__all__ = [
    'Add',
    'Subtract',
    'Multiply',
    'Divide',
    'Negative',
    'Reciprocal',
    'Square',
    'Exponent',
    'Sqrt',
    'Abs',
    'Sum',
    'CumSum',
    'Diff',
    'Roots',
    'Average',
    'Difference',
    'Ratio',
    'Sin',
    'Cos',
    'Tan',
    'ArcSin',
    'ArcCos',
    'ArcTan',
]


class Negative(Transformer):
    def __init__(self, *parents, **kwargs):
        super(Negative, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def apply(self, x):
        return apply(np.negative, x)


class Reciprocal(Transformer):
    def __init__(self, *parents, **kwargs):
        super(Reciprocal, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def apply(self, x):
        return apply(np.reciprocal, x)


class Square(Transformer):
    def __init__(self, *parents, **kwargs):
        super(Square, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def apply(self, x):
        return apply(np.square, x)


class Exponent(Transformer):
    def __init__(self, *parents, a=2, **kwargs):
        super(Exponent, self).__init__(*parents, **kwargs)
        self.a = a
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def apply(self, x):
        def calculator(a):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return np.power(a, self.a)

        return apply(calculator, x)


class Sqrt(Transformer):
    def __init__(self, *parents, **kwargs):
        super(Sqrt, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def apply(self, x):
        return apply(np.sqrt, x)


class Abs(Transformer):
    def __init__(self, *parents, **kwargs):
        super(Abs, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def apply(self, x):
        return apply(np.abs, x)


class Sum(Transformer):
    def __init__(self, *parents, axis=None, **kwargs):
        super(Sum, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def apply(self, x):
        def calculator(a):
            return np.nansum(a, axis=-1, keepdims=True)

        return apply_to_axis(calculator, x, axis=self.axis)


class CumSum(Transformer):
    def __init__(self, *parents, axis=None, **kwargs):
        super(CumSum, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def apply(self, x):
        def calculator(a):
            return np.nancumsum(a, axis=-1)

        return apply_to_axis(calculator, x, axis=self.axis)


class Diff(Transformer):
    def __init__(self, *parents, axis=None, **kwargs):
        super(Diff, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def graph(self, x):
        return Graph(
            Slice(x, i=(1, None), axis=self.axis)
            - Slice(x, i=(0, -1), axis=self.axis)
        )


class Roots(Transformer):
    def __init__(self, *parents, axis=None, **kwargs):
        super(Roots, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def apply(self, x):
        def calculator1d(a):
            return np.roots(a).real

        def calculator(a):
            return np.apply_along_axis(calculator1d, -1, a)

        return apply_to_axis(calculator, x, axis=self.axis)


class Average(Transformer):
    def __init__(self, *parents, rel=False, axis=None, **kwargs):
        super(Average, self).__init__(*parents, **kwargs)
        self.rel = rel
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 2,
            lambda x, y: np.issubdtype(x.dtype, np.float64)
                         and np.issubdtype(y.dtype, np.float64),
        ]

    @staticmethod
    def graph(x, y):
        return Graph(Add(x, y) / Constant(2))


class Difference(Transformer):
    def __init__(self, *parents, rel=False, axis=None, **kwargs):
        super(Difference, self).__init__(*parents, **kwargs)
        self.rel = rel
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 2,
            lambda x, y: np.issubdtype(x.dtype, np.float64) and np.issubdtype(y.dtype, np.float64),
        ]

    def graph(self, x, y):
        if self.rel:
            return Graph(Abs(Subtract(x, y) / x))
        else:
            return Graph(Abs(Subtract(x, y)))


class Ratio(Transformer):
    def __init__(self, *parents, axis=None, **kwargs):
        super(Ratio, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 2,
            lambda x, y: np.issubdtype(x.dtype, np.float64) and np.issubdtype(y.dtype, np.float64),
        ]

    def graph(self, x, y):
        return Graph(x / y)


class Sin(Transformer):
    def __init__(self, *parents, **kwargs):
        super(Sin, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def apply(self, x):
        return apply(np.sin, x)


class Cos(Transformer):
    def __init__(self, *parents, **kwargs):
        super(Cos, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def apply(self, x):
        return apply(np.cos, x)


class Tan(Transformer):
    def __init__(self, *parents, **kwargs):
        super(Tan, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def apply(self, x):
        return apply(np.tan, x)


class ArcSin(Transformer):
    def __init__(self, *parents, **kwargs):
        super(ArcSin, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def apply(self, x):
        result = apply(np.arcsin, x)
        result._unit = units.rad
        return result


class ArcCos(Transformer):
    def __init__(self, *parents, **kwargs):
        super(ArcCos, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def apply(self, x):
        result = apply(np.arccos, x)
        result._unit = units.rad
        return result


class ArcTan(Transformer):
    def __init__(self, *parents, **kwargs):
        super(ArcTan, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def apply(self, x):
        result = apply(np.arctan, x)
        result._unit = units.rad
        return result
