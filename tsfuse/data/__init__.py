import os
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
    Data structure for representing time series data and attribute-value data.

    Internally, collections store data as a three-dimensional arrays of shape
    `(N, t, d)` where:

    - `N` is the number of instances,
    - `t` is the number of time stamps per instance,
    - `d` is the number of dimensions.

    **Representing time series data**

    For time series data, each instance represents a window of `t` time stamps, 
    each having `d` values, i.e., one value per dimension. Hence, each window
    is a multivariate time series with `d` variables and `t` samples per 
    variable.

    Note that the above representation requires a fixed window size `t` for all
    instances. Variable size windows can be represented in two ways:

    - Pad each window to the maximal length using a value `x` and set 
      ``mask_value=x``.
    - If the above is not desired due to memory issues (since padding
      increases the number of values that need to be stored), create one
      collection for each window (with `N = 1`) and initialize a new collection
      with a list of these collections instead of a three-dimensional array.

    **Representing attribute-value data**

    Attribute-value data is represented using the same three-dimensional array.
    However, `t = 1` as attributes have no time dimension.

    Examples
    --------

    The examples below show how to create different kinds of collections.
    See below for more advanced parameters to add instance identifiers, time 
    stamps, dimension names, tags and units.

    .. note::
    
        For convenience, it is also possible to convert a DataFrame to a 
        Collection and vice versa using the :func:`Collection.from_dataframe` 
        and :func:`~Collection.to_dataframe` methods.

    Example 1: Create a time series collection with a fixed window size

    .. code:: python
       
        # N=2 instances with t=5 time stamps and d=3 dimensions
        fixed_window_time_series_collection = Collection([
            # Instance A
            [[1, 5, 10],   # Time stamp 1
             [2, 4, 20],   # Time stamp 2
             [3, 3, 30],   # Time stamp 3
             [4, 2, 40],   # Time stamp 4
             [5, 1, 50]],  # Time stamp 5
            # Instance B
            [[2, 16, 60],  # Time stamp 1
             [4, 14, 70],  # Time stamp 2
             [6, 12, 40],  # Time stamp 3
             [8, 10, 50],  # Time stamp 4
             [10, 8, 20]], # Time stamp 5
        ])

    Example 2: Create a time series collection with a variable window size
    
    .. code:: python
       
        # N=2 instances with d=3 dimensions:
        # one with t=5 time stamps and one with t=4 time stamps

        # Instance A
        a = Collection([
            [[1, 5, 10],   # Time stamp 1
             [2, 4, 20],   # Time stamp 2
             [3, 3, 30],   # Time stamp 3
             [4, 2, 40],   # Time stamp 4
             [5, 1, 50]]   # Time stamp 5
        ])

        # Instance B
        b = Collection([
            [[2, 16, 60],  # Time stamp 1
             [4, 14, 70],  # Time stamp 2
             [6, 12, 40],  # Time stamp 3
             [8, 10, 50]]  # Time stamp 4
        ])

        # One collection with both instances
        variable_window_time_series_collection = Collection([a, b])

    Example 3: Create an attribute collection
    
    .. code:: python
       
        # N=2 instances with d=3 dimensions
        attribute_collection = Collection([
            # Instance A
            [[1, 5, 10]],
            # Instance B
            [[8, 3, 40]],
        ])

    Parameters
    ----------

    values : array-like
        Three-dimensional array of shape `(N, t, d)`.

    index : array-like, optional
        Two-dimensional array of shape `(N, t)` with the time stamps of each 
        window.

    id : array-like, optional
        One-dimensional array of length `N` with the instance identifiers.

    dimensions : array-like, optional
        One-dimensional array of length `d` with the dimension names.

    tags : Tags, optional
        Tags specified by keys and values.

    unit : optional
        Unit, see the `Pint documentation <https://pint.readthedocs.io>`_.

    mask_value : optional
        Placeholder representing missing values.
        Can be used for representing variable-length time series.
        Default: ``nan``
    """

    def __init__(self, values, index=None, id=None, dimensions=None,
                 tags=None, unit=None, mask_value=np.nan):
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
                    collection_id = id[i] if id is not None else i
                    collections.append(Collection(
                        values[i],
                        index=collection_index,
                        id=collection_id,
                        dimensions=dimensions,
                        mask_value=mask_value,
                        unit=self._unit,
                        tags=self._tags
                    ))
            self._values = np.array(collections)
            if index is None:
                self._index = [c.index for c in collections]
            else:
                self._index = index
            if id is None:
                self._id = [c.id for c in collections]
            else:
                self._id = np.array(id, copy=False)
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
            id = np.array(id, copy=False) if id is not None else None
            dimensions = np.array(dimensions, copy=False) if dimensions is not None else None
            values, index, id, dimensions = \
                _reshape(values, index, id, dimensions)
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
            # Determine type of id
            self._idtype = id.dtype
            if np.issubdtype(self._idtype, np.character):
                self._idtype = np.str_
            elif np.issubdtype(self._idtype, np.integer):
                self._idtype = np.int64
            else:
                self._idtype = object
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
            # Index, id and dimensions
            self._index = np.array(index, copy=True, dtype=self._itype)
            self._id = np.array(id, copy=True, dtype=self._idtype)
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
    def id(self):
        """
        numpy.array : One-dimensional array of length ``N``
        """
        return self._id

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

    def to_dataframe(self, column_id='id', column_sort='time'):
        """
        Convert to a DataFrame.
        """
        import pandas as pd
        # Variable length time series collection
        if isinstance(self.shape[1], tuple):
            dataframes = []
            for c in self.values:
                df = pd.DataFrame(c.values.reshape((-1, c.shape[-1])))
                df.columns = c.dimensions
                df.insert(0, column_id, np.repeat(c.id, c.shape[1]))
                df.insert(1, column_sort, c.index.flatten())
                dataframes.append(df)
            return pd.concat(dataframes, axis=0, ignore_index=True)
        # Fixed length time series collection
        elif self.shape[1] > 1:
            df = pd.DataFrame(self.values.reshape((-1, self.shape[-1])))
            df.columns = self.dimensions
            df.insert(0, column_id, np.repeat(self.id, self.shape[1]))
            df.insert(1, column_sort, self.index.flatten())
            return df
        # Attribute collection
        else:
            df = pd.DataFrame(self.values.reshape((-1, self.shape[-1])))
            df.columns = self.dimensions
            df.insert(0, column_id, np.repeat(self.id, self.shape[1]))
            return df

    @classmethod
    def from_dataframe(cls, df, column_id='id', column_sort='time'):
        """
        Create a collection from a DataFrame.

        The given DataFrame can either contain time series data or 
        attribute-value data. For compatibility and ease-of-use, we follow the 
        `tsfresh time  series format (option 1) <https://tsfresh.readthedocs.io/en/latest/text/data_formats.html#input-option-1-flat-dataframe>`_.
        
        For time series data, the following format is expected:

        +-----+------+----------+----------+----------+
        | id  | time | x        | y        | z        |
        +=====+======+==========+==========+==========+
        | A   | t1   | x(A, t1) | y(A, t1) | z(A, t1) |
        +-----+------+----------+----------+----------+
        | A   | t2   | x(A, t2) | y(A, t2) | z(A, t2) |
        +-----+------+----------+----------+----------+
        | ... | ...  | ...      | ...      | ...      |
        +-----+------+----------+----------+----------+
        | B   | t1   | x(B, t1) | y(B, t1) | z(B, t1) |
        +-----+------+----------+----------+----------+
        | B   | t2   | x(B, t2) | y(B, t2) | z(B, t2) |
        +-----+------+----------+----------+----------+
        | ... | ...  | ...      | ...      | ...      |
        +-----+------+----------+----------+----------+

        For attribute-value data, the following format is expected:

        +-----+------+------+------+
        | id  | x    | y    | z    |
        +=====+======+======+======+
        | A   | x(A) | y(A) | z(A) |
        +-----+------+------+------+
        | B   | x(B) | y(B) | z(B) |
        +-----+------+------+------+
        | ... | ...  | ...  | ...  |
        +-----+------+------+------+

        Both time series data and attribute-value data can consist of multiple 
        variables (**x**, **y**, **z**) that have values for multiple instances
        (**id**: A, B, ...). For time series data, each variable represents a 
        time series and hence has multiple values (**time**: t1, t2, ...). The 
        values and the length of the time column do not have to be identical 
        across instances.

        Parameters
        ----------
        df : DataFrame
            Time series data or attribute-value given as a DataFrame in the 
            format specified above.
        column_id : str, optional
            Name of the **id** column. Default: 'id'
        column_sort : str, optional
            Name of the **time** column. Default: 'time'
        """
        # Sort by column_sort, if this column exists
        if column_sort in df.columns: df = df.sort_values(column_sort)
        # Group by column_id and gather some information about the data
        gb = df.groupby(column_id)
        lengths = list(len(gb.groups[g]) for g in gb.groups)
        fixed_length = len(set(lengths)) == 1
        columns = [c for c in df.columns if (c != column_id) and (c != column_sort)]
        dtype = np.find_common_type(list(df.loc[:, columns].dtypes), [])
        # Check if attribute-value data
        if (column_sort not in df.columns) and fixed_length and (lengths[0] == 1):
            shape = (len(gb.groups), 1, len(columns))
            values = np.empty(shape, dtype=dtype)
            id = []
            for i, g in enumerate(gb.groups):
                values[i, :, :] = df.loc[gb.groups[g], columns]
                id.append(g)
            return Collection(values, id=id, dimensions=columns)
        # Check if time series data data
        elif column_sort in df.columns:
            itype = df.dtypes[column_sort]
            # Fixed length time series data: create one collection
            if fixed_length:
                shape = (len(gb.groups), lengths[0], len(columns))
                values = np.empty(shape, dtype=dtype)
                index = np.empty(shape[:2], dtype=itype)
                id = []
                for i, g in enumerate(gb.groups):
                    values[i, :, :] = df.loc[gb.groups[g], columns]
                    index[i, :] = df.loc[gb.groups[g], column_sort]
                    id.append(g)
                return Collection(values, index=index, id=id, dimensions=columns)
            # Variable length time series data: create multiple collections
            else:
                collections = []
                for i, g in enumerate(gb.groups):
                    shape = (1, lengths[i], len(columns))
                    values = np.empty(shape, dtype=dtype)
                    index = np.empty(shape[:2], dtype=itype)
                    values[0, :, :] = df.loc[gb.groups[g], columns]
                    index[0, :] = df.loc[gb.groups[g], column_sort]
                    c = Collection(values, index=index, id=[g], dimensions=columns)
                    collections.append(c)
                return Collection(collections)
        else:
            # TODO: Raise an error
            pass

    def __len__(self):
        return self.shape[0]


def plot(X, i=0):
    """
    Plot time series collections.

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


def _reshape(values, index, id, dimensions):
    if len(values.shape) == 0:
        v = np.empty((1, 1, 1), dtype=values.dtype)
        v[0, 0, 0] = values
        if dimensions is None:
            n = np.zeros((1,))
        else:
            n = np.empty((1,))
            n[0] = dimensions
        if id is None:
            d = np.zeros((1,))
        else:
            d = np.empty((1,))
            d[0] = id
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
        if id is None:
            d = np.zeros((1,))
        else:
            d = id
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
        if id is None:
            d = np.zeros((1,))
        else:
            d = id
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
        if id is None:
            d = np.arange(values.shape[0])
        else:
            d = id
        if index is None:
            t = np.prod(v.shape[:2])
            i = np.empty((v.shape[0], v.shape[1]), dtype=int)
            i[:, :] = np.arange(t).reshape(*v.shape[:2][::-1]).T
        else:
            i = np.empty((v.shape[0], v.shape[1]), dtype=index.dtype)
            i[:, :] = index
    return v, i, d, n


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
