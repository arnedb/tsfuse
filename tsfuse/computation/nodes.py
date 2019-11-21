import abc
import six
import inspect

from functools import partial

import numpy as np
from tsfuse.data import Tags

from ..data import Collection
from ..errors import InvalidPreconditionError


@six.add_metaclass(abc.ABCMeta)
class Node(object):

    def __init__(self, parents=None, is_output=None):
        self._id = None
        self._parents = [] if parents is None else parents
        for p in self._parents:
            p.add_child(self)
        self._children = []
        self._output = None
        # self._output_last = None
        self._is_output = is_output

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, i):
        self._id = i

    @property
    def parents(self):
        """list(Node): Parents of the Node."""
        return self._parents

    @property
    def children(self):
        """list(Node): Children of the Node."""
        return self._children

    @property
    def output(self):
        """Collection: Collection produced by the Node."""
        return self._output

    @output.setter
    def output(self, data):
        """
        Set the output of the Node.

        Parameters
        ----------
        data : DataCollection
            Output of the Node.
        """
        self._output = data

    @property
    def is_input(self):
        """True if the Node is an input of a Graph."""
        return False

    @property
    def is_output(self):
        """True if the node is an output of a graph."""
        if self._is_output is None:
            return len(self._children) == 0
        else:
            return self._is_output

    def add_child(self, child):
        """Add a Node to the children of the Node."""
        self._children.append(child)

    def __add__(self, other):
        return Add(self, other)

    def __sub__(self, other):
        return Subtract(self, other)

    def __mul__(self, other):
        return Multiply(self, other)

    def __div__(self, other):
        return Divide(self, other)

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __truediv__(self, other):
        return self.__div__(other)

    def __gt__(self, other):
        return Greater(self, other)

    def __lt__(self, other):
        return Less(self, other)

    def __le__(self, other):
        return LessEqual(self, other)

    def __ge__(self, other):
        return GreaterEqual(self, other)

    def __neg__(self):
        return Not(self)

    @property
    def trace(self):
        return ()

    @property
    def __name__(self):
        return str(self)

    @property
    def __repr__(self):
        return self.__str__


class Input(Node):
    """
    Node that serves as the Input of a Graph.

    Parameters
    ----------
    input_id : int or str
        Input identifier.
    """

    def __init__(self, input_id):
        super(Input, self).__init__()
        self.input_id = input_id

    @property
    def is_input(self):
        return True

    @property
    def is_output(self):
        return False

    def apply(self, copy=True):
        pass

    @property
    def trace(self):
        return 'Input', self.input_id

    def __str__(self):
        return 'Input({})'.format(self.input_id)


class Constant(Node):
    """
    Node that produces a constant, i.e. a Collection not depending on any
    other Node's output.

    Parameters
    ----------
    data : int, float, str or object
        Output data.
    """

    def __init__(self, data):
        super(Constant, self).__init__()
        self.output = data

    def apply(self, copy=True):
        """Produce the output of the Node and store it in `self.output``."""
        pass

    @property
    def trace(self):
        return 'Constant', self.output

    def __str__(self):
        return 'Constant({})'.format(self.output)


@six.add_metaclass(abc.ABCMeta)
class Transformer(Node):

    def __init__(self, *parents, **kwargs):
        is_output = kwargs.get('is_output', None)
        if not hasattr(self, 'preconditions'):
            self.preconditions = []
        self.preconditions += kwargs.get('with_preconditions', [])
        super(Transformer, self).__init__(parents=parents, is_output=is_output)

    def check_preconditions(self, *collections):
        def satisfied(c):
            return all(p(*c) for p in self.preconditions)

        if isinstance(collections[0].shape[1], tuple):
            for i in range(len(collections[0].shape[1])):
                if not satisfied([c for c in collections]):
                    raise InvalidPreconditionError(self)

        else:
            if not satisfied(collections):
                raise InvalidPreconditionError(self)

    def transform(self, *collections, ignore_preconditions=False):
        if not ignore_preconditions:
            self.check_preconditions(*collections)
        result = None
        if hasattr(self, 'apply'):
            if isinstance(collections[0].shape[1], tuple):
                f = partial(_apply, apply=self.apply, collections=collections[:])
                try:
                    results = [f(i) for i in range(len(collections[0].values))]
                except: # TODO: Make more restrictive!!
                    # TODO: Generate warning instead of error
                    results = None
                if results is not None:
                    if any(r is None for r in results):
                        result = None
                    elif len(set(r.shape for r in results)) == 1:
                        result = Collection(
                            values=np.concatenate([r.values for r in results]),
                            index=np.concatenate([r.index for r in results]),
                            dimensions=results[0].dimensions,
                            mask_value=results[0].mask_value,
                        )
                    else:
                        result = Collection(results)
                else:
                    result = None
            else:
                try:
                    result = self.apply(*collections)
                except:
                    # TODO: Generate warning instead of error
                    result = None
        elif hasattr(self, 'graph'):
            graph = self.graph(*[Input(i) for i in range(len(collections))])
            outputs = graph.transform({i: c for i, c in enumerate(collections)})
            result = outputs[graph.outputs[-1]]
        if result is None:
            return None
        else:
            result._tags = self.tags(*collections)
            result._unit = self.unit(*collections)
            return result

    def tags(self, *collections):
        collections = [c for c in collections if hasattr(c, '_tags')]
        if len(collections) < 1:
            return Tags()
        propagated = Tags(collections[0]._tags)
        for i in range(1, len(collections)):
            propagated = propagated.intersect(collections[i]._tags)
        return propagated

    def unit(self, *collections):
        return None

    @property
    def trace(self):
        def parameter(p):
            if isinstance(p, Transformer):
                return p.trace
            else:
                return p

        values = {p: self.__dict__[p] for p in self.__dict__ if _is_parameter(p)}
        params = tuple([parameter(values[p]) for p in sorted(values)])
        parents = tuple([p.trace for p in self.parents])
        t = tuple([self.__class__.__name__, params, parents])
        return t

    @property
    def n_inputs(self):
        if hasattr(self, 'apply'):
            f = self.apply
        else:
            f = self.graph
        args = inspect.getfullargspec(f)[0]
        return len(args) - 1 if 'self' in args else len(args)

    def __str__(self):
        s = str(self.__class__.__name__)
        values = {p: self.__dict__[p] for p in self.__dict__ if _is_parameter(p)}
        params = sorted(list(values))
        s += '({})'.format(', '.join(
            [str(p) for p in self.parents] +
            ['{}={}'.format(p, values[p]) for p in params if values[p] is not None]
        ))
        return s


def _is_parameter(p):
    if p[0].startswith('_'):
        return False
    else:
        return p not in ('preconditions',)


def _apply(i, apply=None, collections=None):
    inputs = []
    for c in collections:
        if isinstance(c, Collection) and isinstance(c.values[i], Collection):
            inputs.append(c.values[i])
        else:
            inputs.append(c)
    return apply(*inputs)


class Add(Transformer):
    def __init__(self, *parents, **kwargs):
        super(Add, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 2,
            lambda x, y: np.issubdtype(x.dtype, np.float64) & np.issubdtype(x.dtype, np.float64),
        ]

    @staticmethod
    def apply(x, y):
        x, y = _collections(x, y)
        values = x.values + y.values
        return _result(x, y, values)


class Subtract(Transformer):
    def __init__(self, *parents, **kwargs):
        super(Subtract, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 2,
            lambda x, y: np.issubdtype(x.dtype, np.float64) & np.issubdtype(x.dtype, np.float64),
        ]

    @staticmethod
    def apply(x, y):
        x, y = _collections(x, y)
        values = x.values - y.values
        return _result(x, y, values)


class Multiply(Transformer):
    def __init__(self, *parents, **kwargs):
        super(Multiply, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 2,
            lambda x, y: np.issubdtype(x.dtype, np.float64) & np.issubdtype(x.dtype, np.float64),
        ]

    @staticmethod
    def apply(x, y):
        x, y = _collections(x, y)
        values = x.values * y.values
        return _result(x, y, values)


class Divide(Transformer):
    def __init__(self, *parents, **kwargs):
        super(Divide, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 2,
            lambda x, y: np.issubdtype(x.dtype, np.float64) & np.issubdtype(x.dtype, np.float64),
        ]

    @staticmethod
    def apply(x, y):
        x, y = _collections(x, y)
        with np.errstate(divide='ignore', invalid='ignore'):
            values = np.true_divide(x.values, y.values)
        return _result(x, y, values)


class Greater(Transformer):
    def __init__(self, *parents, **kwargs):
        super(Greater, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 2,
            lambda x, y: np.issubdtype(x.dtype, np.float64) & np.issubdtype(x.dtype, np.float64),
        ]

    @staticmethod
    def apply(x, y):
        x, y = _collections(x, y)
        values = np.array(x.values > y.values, dtype=bool)
        return _result(x, y, values)


class GreaterEqual(Transformer):
    def __init__(self, *parents, **kwargs):
        super(GreaterEqual, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 2,
            lambda x, y: np.issubdtype(x.dtype, np.float64) & np.issubdtype(x.dtype, np.float64),
        ]

    @staticmethod
    def apply(x, y):
        x, y = _collections(x, y)
        values = np.array(x.values >= y.values, dtype=bool)
        return _result(x, y, values)


class Less(Transformer):
    def __init__(self, *parents, **kwargs):
        super(Less, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 2,
            lambda x, y: np.issubdtype(x.dtype, np.float64) & np.issubdtype(x.dtype, np.float64),
        ]

    @staticmethod
    def apply(x, y):
        x, y = _collections(x, y)
        values = np.array(x.values < y.values, dtype=bool)
        return _result(x, y, values)


class LessEqual(Transformer):
    def __init__(self, *parents, **kwargs):
        super(LessEqual, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 2,
            lambda x, y: np.issubdtype(x.dtype, np.float64) & np.issubdtype(x.dtype, np.float64),
        ]

    @staticmethod
    def apply(x, y):
        x, y = _collections(x, y)
        values = np.array(x.values <= y.values, dtype=bool)
        return _result(x, y, values)


class And(Transformer):
    def __init__(self, *parents, **kwargs):
        super(And, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 2,
            lambda x, y: np.issubdtype(x.dtype, np.bool_) & np.issubdtype(y.dtype, np.bool_),
        ]

    @staticmethod
    def apply(x, y):
        x, y = _collections(x, y)
        values = np.logical_and(x.values, y.values)
        return _result(x, y, values)


class Or(Transformer):
    def __init__(self, *parents, **kwargs):
        super(Or, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 2,
            lambda x, y: np.issubdtype(x.dtype, np.bool_) & np.issubdtype(y.dtype, np.bool_),
        ]

    @staticmethod
    def apply(x, y):
        x, y = _collections(x, y)
        values = np.logical_or(x.values, y.values)
        return _result(x, y, values)


class Not(Transformer):
    def __init__(self, *parents, **kwargs):
        super(Not, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.bool_),
        ]

    @staticmethod
    def apply(x):
        values = np.logical_not(x.values)
        return Collection(values)


def _collections(x, y):
    if not isinstance(x, Collection):
        x = Collection(x)
    if not isinstance(y, Collection):
        y = Collection(y)
    return x, y


def _result(x, y, values):
    if values.shape == x.shape:
        return Collection(values, index=x.index, dimensions=x.dimensions)
    else:
        return Collection(values, index=y.index, dimensions=y.dimensions)
