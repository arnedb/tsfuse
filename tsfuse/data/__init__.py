from enum import Enum

import numpy as np

from .tags import Tags, TagKey, HierarchicalTagKey
from .units import units

__all__ = [
    'Collection',
    'Type',
]


class Collection(object):
    """
    Data collection for representing time series and attributes.

    Parameters
    ----------

    values : array-like
        Three-dimensional array of shape ``(N, t, d)`` where:

        - `N` is the number of windows
        - `t` is the number of time stamps in each window
        - `d` is the number of dimensions.

    index : array-like, optional
        Two-dimensional array of shape ``(N, t)`` with the time stamps of each window.

    dimensions : array-like, optional
        One-dimensional array of length ``d`` with the dimension names.

    mask_value : optional
        Placeholder representing missing values.
        Can be used for representing variable-length time series.
        Default: ``nan``

    unit : optional
        Unit, see the `Pint documentation <https://pint.readthedocs.io>`_.

    tags : optional
        :class:`~tsfuse.data.tags.Tags` object which specifies values for one or more tag keys.
    """

    def __init__(self, values, index=None, dimensions=None,
                 mask_value=np.nan, unit=None, tags=None):
        # Unit
        self._unit = unit
        # Tags
        self._tags = tags if tags is not None else Tags()
        # Data
        # TODO: Add checks for shapes
        if _collections(values):
            collections = []
            for i in range(len(values)):
                if values[i].__class__.__name__ == 'Collection':
                    collections.append(values[i])
                else:
                    collection_index = index[i] if index is not None else None
                    collections.append(Collection(
                        values[i],
                        index=collection_index,
                        dimensions=dimensions,
                        mask_value=mask_value,
                        unit=self._unit,
                        tags=self._tags
                    ))
            self._values = np.array(collections)
            if index is None:
                self._index = np.arange(np.sum([c.shape[0] for c in collections]))
            else:
                self._index = index
            if dimensions is None:
                if len(set(tuple(c.dimensions) for c in collections)) == 1:
                    self._dimensions = collections[0].dimensions
                else:
                    self._dimensions = np.arange(collections[0].shape[2])
            else:
                self._dimensions = np.array(dimensions, copy=False)
        else:
            values = np.array(values, copy=False)
            index = np.array(index, copy=False) if index is not None else None
            dimensions = np.array(dimensions, copy=False) if dimensions is not None else None
            values, index, dimensions = _reshape(values, index, dimensions)
            # Determine data type of values
            self._dtype = values.dtype
            if np.issubdtype(self._dtype, np.number):
                self._dtype = np.float64
            elif np.issubdtype(self._dtype, np.character):
                self._dtype = np.str_
            elif np.issubdtype(self._dtype, np.bool_):
                self._dtype = np.bool_
            else:
                self._dtype = object
            # Determine data type of index
            self._itype = index.dtype
            if np.issubdtype(self._itype, np.datetime64):
                self._itype = np.dtype('datetime64[ns]')
            elif np.issubdtype(self._itype, np.integer):
                self._itype = np.int64
            else:
                self._itype = object
            # Determine type of dimensions
            self._ntype = dimensions.dtype
            if np.issubdtype(self._ntype, np.character):
                self._ntype = np.str_
            elif np.issubdtype(self._ntype, np.integer):
                self._ntype = np.int64
            else:
                self._ntype = object
            # Values
            self._values = np.empty(values.shape, dtype=self._dtype)
            self._values[:, :, :] = values
            # Index and dimensions
            self._index = np.array(index, copy=True, dtype=self._itype)
            self._dimensions = np.array(dimensions, copy=True, dtype=self._ntype)
        # Mask value
        self._mask_value = mask_value

    @property
    def values(self):
        """
        numpy.array : Three-dimensional array of shape ``(N, t, d)``
        """
        return self._values

    @property
    def index(self):
        """
        numpy.array : Two-dimensional array of shape ``(N, t)``
        """
        return self._index

    @property
    def dimensions(self):
        """
        numpy.array : One-dimensional array of length ``d``
        """
        return self._dimensions

    @property
    def tags(self):
        """
        :class:`~tsfuse.data.tags.Tags` object which specifies values for one or more tag keys.
        """
        return self._tags

    @property
    def unit(self):
        """
        pint.unit.Unit : Unit of ``self.values``
        """
        return self._unit

    @property
    def type(self):
        """
        tsfuse.data.Type : Type of data.
        """
        if self.shape[0] > 1:
            return Type.WINDOWS
        elif np.max(self.shape[1]) > 1:
            return Type.SERIES
        elif self.shape[2] > 1:
            return Type.ATTRIBUTES
        else:
            return Type.SCALAR

    @property
    def transform_axis(self):
        if (self.type == Type.WINDOWS) or (self.type == Type.SERIES):
            return 'timestamps'
        else:
            return 'dimensions'

    @property
    def mask_value(self):
        """
        Mask value.
        """
        return self._mask_value

    @property
    def shape(self):
        """
        Shape ``(N, t, d)``
        """
        if np.isscalar(self.values):
            return ()
        elif len(self.values.shape) == 1:
            return (
                np.sum([c.shape[0] for c in self.values]),
                tuple([c.shape[1] for c in self.values]),
                len(self.dimensions)
            )
        else:
            return self.values.shape

    @property
    def loc(self):
        return NotImplementedError()

    @property
    def iloc(self):
        return IndexLocationIndexer(self)

    @property
    def dtype(self):
        """
        Data type of ``self.values``
        """
        if (self.values is not None) and (len(self.values.shape) == 3):
            return np.array(self.values, copy=False).dtype
        else:
            return None

    @property
    def itype(self):
        """
        Data type of ``self.index``
        """
        if (self.values is not None) and (len(self.index.shape) == 2):
            return self.index.dtype
        else:
            return None

    def append(self, other):
        values = np.concatenate((self.values, other.values), axis=0)
        index = np.concatenate((self.index, other.index), axis=0)
        dimensions = self.dimensions
        m = self.mask_value
        return Collection(values, dimensions=dimensions, index=index, mask_value=m)

    def __len__(self):
        return self.shape[0]


def plot(X, i=0):
    """Plot time series collections.

    Parameters
    ----------
    X : dict
        Time series collections.
    i : int, optional
        Select window ``i`` using ``.values[i, :, :]``. Default: 0
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 2 * len(X)))
    for v, name in enumerate(X):
        plt.subplot(len(X), 1, v + 1)
        collection = X[name]
        for j in range(collection.shape[2]):
            # TODO: Fix index of Collection
            plt.plot(X[name].index[i, :], X[name].values[i, :, j].flatten())
        plt.ylabel(name)
    plt.show()


class IndexLocationIndexer(object):
    def __init__(self, collection):
        self.collection = collection

    def __getitem__(self, item):
        values = self.collection.values[item]
        dimensions = self.collection.dimensions[item[-1]]
        index = self.collection.index[item[:-1]]
        m = self.collection.mask_value
        return Collection(values, dimensions=dimensions, index=index, mask_value=m)


class Type(Enum):
    """
    Type of data.
    """

    SCALAR = 0
    """
    One value (0-dimensional)
    """

    ATTRIBUTES = 1
    """
    Vector of attributes (1-dimensional)
    """

    SERIES = 2
    """
    Vector of one or more time series (2-dimensional)
    """

    WINDOWS = 3
    """
    Windows, each with one or more time series (3-dimensional)
    """


def _reshape(values, index, dimensions):
    if len(values.shape) == 0:
        v = np.empty((1, 1, 1), dtype=values.dtype)
        v[0, 0, 0] = values
        if dimensions is None:
            n = np.zeros((1,))
        else:
            n = np.empty((1,))
            n[0] = dimensions
        if index is None:
            i = np.zeros((1, 1))
        else:
            i = np.empty((1, 1))
            i[0, 0] = index
    elif len(values.shape) == 1:
        v = np.empty((1, 1, values.shape[0]), dtype=values.dtype)
        v[0, 0, :] = values
        if dimensions is None:
            n = np.arange(v.shape[2])
        else:
            n = dimensions
        if index is None:
            i = np.zeros((1, 1))
        else:
            i = np.empty((1, 1))
            i[0, 0] = index
    elif len(values.shape) == 2:
        v = np.empty((1, values.shape[0], values.shape[1]), dtype=values.dtype)
        v[0, :, :] = values
        if dimensions is None:
            n = np.arange(v.shape[2])
        else:
            n = dimensions
        if index is None:
            i = np.empty((1, v.shape[1]))
            i[0, :] = np.arange(v.shape[1])
        else:
            i = np.empty((1, v.shape[1]), dtype=index.dtype)
            i[0, :] = index
    else:
        v = np.empty((values.shape[0], values.shape[1], values.shape[2]), dtype=values.dtype)
        v[:, :, :] = values
        if dimensions is None:
            n = np.arange(v.shape[2])
        else:
            n = dimensions
        if index is None:
            t = np.prod(v.shape[:2])
            i = np.empty((v.shape[0], v.shape[1]))
            i[:, :] = np.arange(t).reshape(*v.shape[:2][::-1]).T
        else:
            i = np.empty((v.shape[0], v.shape[1]), dtype=index.dtype)
            i[:, :] = index
    return v, i, n


def _collections(values):
    if isinstance(values, (int, float, str)):
        return False
    if len(values) == 0:
        return False
    if isinstance(values[0], (list, tuple, np.ndarray)) \
            and (len(set(len(values[i]) for i in range(len(values)))) > 1):
        return True
    if values[0].__class__.__name__ == 'Collection':
        return True
    return False
