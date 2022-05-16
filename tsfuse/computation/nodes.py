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
    """
    Node of a computation graph.

    Parameters
    ----------
    parents : list(Node), optional
        Parent nodes.
    is_output : bool, optional
        True if the node must be an output node or False if the node should not be an output node.
        By default, the node is an output node if it is not used as a parent for another node.

    Attributes
    ----------
    parents : list(Node)
        Parent nodes.
    children : list(Node)
        Child nodes.
    is_input : bool
        True if the node is an input node.
    is_output : bool
        True if the node is an output node.
    """

    def __init__(self, parents=None, is_output=None):
        self._id = None
        self._parents = [] if parents is None else parents
        for p in self._parents:
            p.add_child(self)
        self._children = []
        self._output = None
        self._is_output = is_output

    @property
    def parents(self):
        return self._parents

    @property
    def children(self):
        return self._children

    @property
    def is_input(self):
        return False

    @property
    def is_output(self):
        if self._is_output is None:
            return len(self._children) == 0
        else:
            return self._is_output

    def add_child(self, child):
        """
        Add a child node.

        Parameters
        ----------
        child : Node
            Child node.
        """
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
    Node that serves as the input of a computation graph.

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
        return "Input", self.input_id

    @property
    def name(self):
        return str(self.input_id)

    def __str__(self):
        return "Input({})".format(self.input_id)


class Constant(Node):
    """
    Node that produces a constant value,
    given as :class:`~tsfuse.data.Collection` object.

    Parameters
    ----------
    data : int, float, str or object
        Output data.
    """

    def __init__(self, data):
        super(Constant, self).__init__()
        self.output = data

    def apply(self):
        pass

    @property
    def trace(self):
        return "Constant", self.output

    @property
    def name(self):
        return "Constant"

    def __str__(self):
        return "Constant({})".format(self.output)


@six.add_metaclass(abc.ABCMeta)
class Transformer(Node):
    """
    Transformer node.


    """

    def __init__(self, *parents, **kwargs):
        is_output = kwargs.get("is_output", None)
        if not hasattr(self, "preconditions"):
            self.preconditions = []
        self.preconditions += kwargs.get("with_preconditions", [])
        super(Transformer, self).__init__(parents=parents, is_output=is_output)

    def check_preconditions(self, *collections):
        """
        Check that the preconditions are satisfied.

        Parameters
        ----------
        *collections
            :class:`~tsfuse.data.Collection` objects used as input.

        Returns
        -------
        satisfied : bool
        """

        def satisfied(c):
            return all(p(*c) for p in self.preconditions)

        if any(c is None for c in collections):
            raise InvalidPreconditionError(self)

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
        if hasattr(self, "apply"):
            if isinstance(collections[0].shape[1], tuple):
                f = partial(
                    _apply, apply=self.apply, collections=collections[:]
                )
                try:
                    results = [f(i) for i in range(len(collections[0].values))]
                except:  # TODO: Make more restrictive!!
                    # TODO: Generate warning instead of error
                    results = None
                if results is not None:
                    if any(r is None for r in results):
                        result = None
                    elif len(set(r.shape for r in results)) == 1:
                        result = Collection.from_array(
                            np.concatenate([r.values for r in results]),
                            time=np.concatenate([r.time for r in results]),
                            dims=results[0].dims,
                        )
                    else:
                        result = Collection.from_array(results)
                else:
                    result = None
            else:
                try:
                    result = self.apply(*collections)
                except:
                    # TODO: Generate warning instead of error
                    result = None
        elif hasattr(self, "graph"):
            graph = self.graph(*[Input(i) for i in range(len(collections))])
            outputs = graph.transform(
                {i: c for i, c in enumerate(collections)},
                return_dataframe=False,
            )
            result = outputs[graph.outputs[-1]]
        if result is None:
            return None
        else:
            result._tags = self.tags(*collections)
            result._unit = self.unit(*collections)
            return result

    def tags(self, *collections):
        collections = [c for c in collections if hasattr(c, "_tags")]
        if len(collections) < 1:
            return Tags()
        propagated = Tags(collections[0]._tags)
        for i in range(1, len(collections)):
            propagated = propagated.intersect(collections[i]._tags)
        return propagated

    def unit(self, *collections):
        pass

    @property
    def trace(self):
        def parameter(p):
            if isinstance(p, Transformer):
                return p.trace
            else:
                return p

        values = {
            p: self.__dict__[p] for p in self.__dict__ if _is_parameter(p)
        }
        params = tuple([parameter(values[p]) for p in sorted(values)])
        parents = tuple([p.trace for p in self.parents])
        t = tuple([self.__class__.__name__, params, parents])
        return t

    @property
    def n_inputs(self):
        if hasattr(self, "apply"):
            f = self.apply
        else:
            f = self.graph
        args = inspect.getfullargspec(f)[0]
        return len(args) - 1 if "self" in args else len(args)

    def __str__(self):
        s = str(self.__class__.__name__)
        values = {
            p: self.__dict__[p] for p in self.__dict__ if _is_parameter(p)
        }
        params = sorted(list(values))
        s += "({})".format(
            ", ".join(
                [str(p) for p in self.parents]
                + [
                    "{}={}".format(p, values[p])
                    for p in params
                    if values[p] is not None
                ]
            )
        )
        return s

    @property
    def name(self):
        args = [
            a
            for a in list(self.__dict__)
            if a
            not in [
                "preconditions",
                "_id",
                "_parents",
                "_children",
                "_output",
                "_is_output",
            ]
        ]
        values = [getattr(self, a) for a in args]
        argsvalues = [(a, v) for a, v in zip(args, values) if v is not None]
        if len(argsvalues) > 0:
            parameters = (
                "("
                + ", ".join(["{}={}".format(a, v) for a, v in argsvalues])
                + ")"
            )
        else:
            parameters = ""
        return str(self.__class__.__name__) + parameters


def _is_parameter(p):
    if p[0].startswith("_"):
        return False
    else:
        return p not in ("preconditions",)


def _apply(i, apply=None, collections=None):
    inputs = []
    for c in collections:
        if isinstance(c, Collection) and isinstance(c.values[i], Collection):
            inputs.append(c.values[i])
        else:
            inputs.append(c)
    return apply(*inputs)


class Add(Transformer):
    """
    Element-wise addition
    """

    def __init__(self, *parents, **kwargs):
        super(Add, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 2,
            lambda x, y: np.issubdtype(x.dtype, np.float64)
            & np.issubdtype(x.dtype, np.float64),
        ]

    def transform(self, x, y, **kwargs):
        """
        Compute :math:`x + y`
        """
        return super().transform(x, y, **kwargs)

    @staticmethod
    def apply(x, y):
        x, y = _collections(x, y)
        values = x.values + y.values
        return _result(x, y, values)


class Subtract(Transformer):
    """
    Element-wise subtraction
    """

    def __init__(self, *parents, **kwargs):
        super(Subtract, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 2,
            lambda x, y: np.issubdtype(x.dtype, np.float64)
            & np.issubdtype(x.dtype, np.float64),
        ]

    def transform(self, x, y, **kwargs):
        """
        Compute :math:`x - y`
        """
        return super().transform(x, y, **kwargs)

    @staticmethod
    def apply(x, y):
        x, y = _collections(x, y)
        values = x.values - y.values
        return _result(x, y, values)


class Multiply(Transformer):
    """
    Element-wise multiplication
    """

    def __init__(self, *parents, **kwargs):
        super(Multiply, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 2,
            lambda x, y: np.issubdtype(x.dtype, np.float64)
            & np.issubdtype(x.dtype, np.float64),
        ]

    def transform(self, x, y, **kwargs):
        """
        Compute :math:`x \\cdot y`
        """
        return super().transform(x, y, **kwargs)

    @staticmethod
    def apply(x, y):
        x, y = _collections(x, y)
        values = x.values * y.values
        return _result(x, y, values)


class Divide(Transformer):
    """
    Element-wise division
    """

    def __init__(self, *parents, **kwargs):
        super(Divide, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 2,
            lambda x, y: np.issubdtype(x.dtype, np.float64)
            & np.issubdtype(x.dtype, np.float64),
        ]

    def transform(self, x, y, **kwargs):
        """
        Compute :math:`x / y`
        """
        return super().transform(x, y, **kwargs)

    @staticmethod
    def apply(x, y):
        x, y = _collections(x, y)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.true_divide(x.values, y.values)
        return _result(x, y, values)


class Greater(Transformer):
    """
    Element-wise greater than comparison
    """

    def __init__(self, *parents, **kwargs):
        super(Greater, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 2,
            lambda x, y: np.issubdtype(x.dtype, np.float64)
            & np.issubdtype(x.dtype, np.float64),
        ]

    def transform(self, x, y, **kwargs):
        """
        Compute :math:`x > y`
        """
        return super().transform(x, y, **kwargs)

    @staticmethod
    def apply(x, y):
        x, y = _collections(x, y)
        values = np.array(x.values > y.values, dtype=bool)
        return _result(x, y, values)


class GreaterEqual(Transformer):
    """
    Element-wise greater than or equal comparison.
    """

    def __init__(self, *parents, **kwargs):
        super(GreaterEqual, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 2,
            lambda x, y: np.issubdtype(x.dtype, np.float64)
            & np.issubdtype(x.dtype, np.float64),
        ]

    def transform(self, x, y, **kwargs):
        """
        Compute :math:`x \\geq y`
        """
        return super().transform(x, y, **kwargs)

    @staticmethod
    def apply(x, y):
        x, y = _collections(x, y)
        values = np.array(x.values >= y.values, dtype=bool)
        return _result(x, y, values)


class Less(Transformer):
    """
    Element-wise less than comparison
    """

    def __init__(self, *parents, **kwargs):
        super(Less, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 2,
            lambda x, y: np.issubdtype(x.dtype, np.float64)
            & np.issubdtype(x.dtype, np.float64),
        ]

    def transform(self, x, y, **kwargs):
        """
        Compute :math:`x < y`
        """
        return super().transform(x, y, **kwargs)

    @staticmethod
    def apply(x, y):
        x, y = _collections(x, y)
        values = np.array(x.values < y.values, dtype=bool)
        return _result(x, y, values)


class LessEqual(Transformer):
    """
    Element-wise less than or equal comparison
    """

    def __init__(self, *parents, **kwargs):
        super(LessEqual, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 2,
            lambda x, y: np.issubdtype(x.dtype, np.float64)
            & np.issubdtype(x.dtype, np.float64),
        ]

    def transform(self, x, y, **kwargs):
        """
        Compute :math:`x \\leq y`
        """
        return super().transform(x, y, **kwargs)

    @staticmethod
    def apply(x, y):
        x, y = _collections(x, y)
        values = np.array(x.values <= y.values, dtype=bool)
        return _result(x, y, values)


class And(Transformer):
    """
    Element-wise logical and
    """

    def __init__(self, *parents, **kwargs):
        super(And, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 2,
            lambda x, y: np.issubdtype(x.dtype, np.bool_)
            & np.issubdtype(y.dtype, np.bool_),
        ]

    def transform(self, x, y, **kwargs):
        """
        Compute :math:`x \\land y`

        Parameters
        ----------
        x : Collection
            Boolean data.
        y : Collection
            Boolean data.
        """
        return super().transform(x, y, **kwargs)

    @staticmethod
    def apply(x, y):
        x, y = _collections(x, y)
        values = np.logical_and(x.values, y.values)
        return _result(x, y, values)


class Or(Transformer):
    """
    Element-wise logical or
    """

    def __init__(self, *parents, **kwargs):
        super(Or, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 2,
            lambda x, y: np.issubdtype(x.dtype, np.bool_)
            & np.issubdtype(y.dtype, np.bool_),
        ]

    def transform(self, x, y, **kwargs):
        """
        Compute :math:`x \\lor y`

        Parameters
        ----------
        x : Collection
            Boolean data.
        y : Collection
            Boolean data.
        """
        return super().transform(x, y, **kwargs)

    @staticmethod
    def apply(x, y):
        x, y = _collections(x, y)
        values = np.logical_or(x.values, y.values)
        return _result(x, y, values)


class Not(Transformer):
    """
    Element-wise logical negation
    """

    def __init__(self, *parents, **kwargs):
        super(Not, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.bool_),
        ]

    def transform(self, x, y, **kwargs):
        """
        Compute :math:`\\neg x`

        Parameters
        ----------
        x : Collection
            Boolean data.
        """
        return super().transform(x, y, **kwargs)

    @staticmethod
    def apply(x):
        values = np.logical_not(x.values)
        return Collection.from_array(values)


def _collections(x, y):
    if not isinstance(x, Collection):
        x = Collection.from_array(x)
    if not isinstance(y, Collection):
        y = Collection.from_array(y)
    return x, y


def _result(x, y, values):
    if values.shape == x.shape:
        return Collection.from_array(values, time=x.time, dims=x.dims)
    else:
        return Collection.from_array(values, time=y.time, dims=y.dims)
