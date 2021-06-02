from enum import Enum

import numpy as np
import pandas as pd

from .tags import Tags
from .units import units

__all__ = [
    'Collection',
    'Type',
]


def create_dataset(X, y=None, column_id='id', column_sort='time', collections=None):
    """
    Create a dataset from given time series data (required) and labels (optional).

    Parameters
    ----------
    X : DataFrame
        Time series data.
    y : Series, optional
        Labels.
    column_id : str, optional
        Name of the column that contains the instance identifiers. Default: 'id'
    column_sort : str, optional
        Name of the column that contains the timestamps. Default: 'time'
    collections : dict, optional
        Dictonary where each key is the names of a collection
        and each value is a list of time series names (i.e., columns of `X`)
        that belong to the collection.
    """
    # Input data checks
    if not isinstance(X, pd.DataFrame):
        raise ValueError('X must be a DataFrame')
    if column_id not in X.columns:
        raise ValueError(f'Column "{column_id}" does not exist in X')
    if column_sort not in X.columns:
        raise ValueError(f'Column "{column_sort}" does not exist in X')
    if y is not None:
        X_ids = X['id'].unique()
        y_ids = y.index
        if not all(i in y_ids for i in X_ids):
            raise ValueError('Not all identifiers in X are in y')
        if not all(i in X_ids for i in y_ids):
            raise ValueError('Not all identifiers in y are in X')
    if collections is not None:
        for c in collections:
            names = collections[c]
            if not isinstance(names, (list, tuple, np.ndarray)):
                raise ValueError(
                    'Column names of each collection must be given as a list'
                )
            for n in names:
                if n not in X.columns:
                    raise ValueError(f'Column "{n}" does not exist in X')
    else:
        ignore_columns = [column_id, column_sort]
        collections = {c: [c] for c in X.columns if c not in ignore_columns}
    
    # Create TSFuse format
    X_tsfuse = dict()
    for c in collections:
        x = X[[column_id, column_sort] + list(collections[c])]
        x[column_id] = x[column_id].astype('category')
        x[column_id].cat.set_categories(y.index, inplace=True)
        x = x.sort_values('id')
        X_tsfuse[c] = Collection(x, column_id=column_id, column_sort=column_sort)

    return X_tsfuse, y


class Collection(object):
    """
    Data structure for representing time series data and attribute-value data.

    Parameters
    ----------
    x : pandas.DataFrame
        Time series or attributes given as a DataFrame.
    column_id : str, optional
        Name of the column that contains the instance identifiers. Default: 'id'
    column_sort : str, optional
        Name of the column that contains the timestamps. Default: 'time'
    tags : optional
        Tags specified by keys and values.
    unit : optional
        Unit, see the `Pint documentation <https://pint.readthedocs.io>`_.
    """

    def __init__(self, x, column_id='id', column_sort='time', tags=None, unit=None):
        # Data
        if isinstance(x, pd.DataFrame):
            self._init_from_dataframe(x, column_id, column_sort)
        elif x is not None:
            raise ValueError('Data must be given as a DataFrame')
        # Tags
        self._tags = tags if tags is not None else Tags()
        # Unit
        self._unit = unit

    @classmethod
    def from_array(cls, values, id=None, time=None, dims=None, tags=None, unit=None):
        collection = Collection(None, tags=tags, unit=unit)
        collection._init_from_array(values, id, time, dims)
        return collection

    def _init_from_dataframe(self, x, column_id, column_sort):
        if column_id not in x.columns:
            raise ValueError(f'Column "{column_id}" does not exist')
        # Parse collection
        parsed_id = x[column_id].unique()
        parsed_data = []
        if len(parsed_id) == len(x):
            # Attribute collection
            parsed_dims = [c for c in x.columns if (c != column_id)]
            for i in parsed_id:
                parsed_data.append(x.loc[x[column_id] == i, parsed_dims].values)
            parsed_data = np.array(parsed_data)
            self._init_from_array(parsed_data, parsed_id, None, parsed_dims)
        else:
            # Time series collection
            if column_sort not in x.columns:
                raise ValueError(f'Column "{column_sort}" does not exist')
            parsed_time = []
            parsed_dims = [c for c in x.columns if (c != column_id) and (c != column_sort)]
            for i in parsed_id:
                parsed_data.append(x.loc[x[column_id] == i, parsed_dims].values)
                parsed_time.append(x.loc[x[column_id] == i, column_sort].values)
            self._init_from_array(parsed_data, parsed_id, parsed_time, parsed_dims)

    def _init_from_array(self, data, id, time, dims):
        # Multiple variable-length collections
        if _multiple_collections(data):
            # Get all collections
            collections = []
            for i in range(len(data)):
                if data[i].__class__.__name__ == 'Collection':
                    collections.append(data[i])
                else:
                    collections.append(Collection.from_array(
                        data[i],
                        id=id[i] if id is not None else i,
                        time=time[i] if time is not None else None,
                        dims=dims
                    ))
            # Store all collections in self._values
            self._values = np.array(collections)
            # Create id property
            if id is None:
                self._id = [c.id for c in collections]
            else:
                self._id = np.array(id, copy=False)
            # Create time property
            if time is None:
                self._time = [c.time for c in collections]
            else:
                self._time = time
            # Create dims property
            if dims is None:
                if len(set(tuple(c.dims) for c in collections)) == 1:
                    self._dims = collections[0].dims
                else:
                    self._dims = np.arange(collections[0].shape[2])
            else:
                self._dims = np.array(dims, copy=False)
        # Single fixed-length collection
        else:
            data = np.array(data, copy=False)
            id = np.array(id, copy=False) if id is not None else None
            time = np.array(time, copy=False) if time is not None else None
            dims = np.array(dims, copy=False) if dims is not None else None
            data, id, time, dims = _reshape(data, id, time, dims)
            # Determine data type of values
            self._dtype = data.dtype
            if np.issubdtype(self._dtype, np.number):
                self._dtype = np.float64
            elif np.issubdtype(self._dtype, np.character):
                self._dtype = np.str_
            elif np.issubdtype(self._dtype, np.bool_):
                self._dtype = np.bool_
            else:
                self._dtype = object
            # Determine type of id
            self._idtype = id.dtype
            if np.issubdtype(self._idtype, np.character):
                self._idtype = np.str_
            elif np.issubdtype(self._idtype, np.integer):
                self._idtype = np.int64
            else:
                self._idtype = object
            # Determine data type of time
            self._itype = time.dtype
            if np.issubdtype(self._itype, np.datetime64):
                self._itype = np.dtype('datetime64[ns]')
            elif np.issubdtype(self._itype, np.integer):
                self._itype = np.int64
            else:
                self._itype = object
            # Determine type of dims
            self._ntype = dims.dtype
            if np.issubdtype(self._ntype, np.character):
                self._ntype = np.str_
            elif np.issubdtype(self._ntype, np.integer):
                self._ntype = np.int64
            else:
                self._ntype = object
            # Values
            self._values = np.empty(data.shape, dtype=self._dtype)
            self._values[:, :, :] = data
            # Set id, time, and dims
            self._id = np.array(id, copy=True, dtype=self._idtype)
            self._time = np.array(time, copy=True, dtype=self._itype)
            self._dims = np.array(dims, copy=True, dtype=self._ntype)

    @property
    def values(self):
        """
        numpy.array : Three-dimensional array of shape ``(N, t, d)``
        """
        return self._values

    @property
    def id(self):
        """
        numpy.array : One-dimensional array of length ``N``
        """
        return self._id

    @property
    def time(self):
        """
        numpy.array : Two-dimensional array of shape ``(N, t)``
        """
        return self._time

    @property
    def dims(self):
        """
        numpy.array : One-dimensional array of length ``d``
        """
        return self._dims

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
            return 'time'
        else:
            return 'dims'

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
                len(self.dims)
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
        Data type of ``self.time``
        """
        if (self.values is not None) and (len(self.time.shape) == 2):
            return self.time.dtype
        else:
            return None

    def to_dataframe(self, id='id', time='time'):
        """
        Convert to a DataFrame.
        """
        # Variable length time series collection
        if isinstance(self.shape[1], tuple):
            dataframes = []
            for c in self.values:
                df = pd.DataFrame(c.values.reshape((-1, c.shape[-1])))
                df.columns = c.dims
                df.insert(0, id, np.repeat(c.id, c.shape[1]))
                df.insert(1, time, c.time.flatten())
                dataframes.append(df)
            return pd.concat(dataframes, axis=0, ignore_index=True)
        # Fixed length time series collection
        elif self.shape[1] > 1:
            df = pd.DataFrame(self.values.reshape((-1, self.shape[-1])))
            df.columns = self.dims
            df.insert(0, id, np.repeat(self.id, self.shape[1]))
            df.insert(1, time, self.time.flatten())
            return df
        # Attribute collection
        else:
            df = pd.DataFrame(self.values.reshape((-1, self.shape[-1])))
            df.columns = self.dims
            print(df.shape, np.repeat(self.id, self.shape[1]).shape)
            df.insert(0, id, np.repeat(self.id, self.shape[1]))
            return df

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
            # TODO: Fix time of Collection
            plt.plot(X[name].time[i, :], X[name].values[i, :, j].flatten())
        plt.ylabel(name)
    plt.show()


class IndexLocationIndexer(object):
    def __init__(self, collection):
        self.collection = collection

    def __getitem__(self, item):
        values = self.collection.values[item]
        dims = self.collection.dims[item[-1]]
        time = self.collection.time[item[:-1]]
        m = self.collection.mask_value
        return Collection(values, dims=dims, time=time, mask_value=m)


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
    Windows, each with one or more attributes or time series (3-dimensional)
    """


def _reshape(data, id, time, dims):
    if len(data.shape) == 0:
        v = np.empty((1, 1, 1), dtype=data.dtype)
        v[0, 0, 0] = data
        if dims is None:
            n = np.zeros((1,))
        else:
            n = np.empty((1,))
            n[0] = dims
        if id is None:
            d = np.zeros((1,))
        else:
            d = np.empty((1,))
            d[0] = id
        if time is None:
            i = np.zeros((1, 1))
        else:
            i = np.empty((1, 1))
            i[0, 0] = time
    elif len(data.shape) == 1:
        v = np.empty((1, 1, data.shape[0]), dtype=data.dtype)
        v[0, 0, :] = data
        if dims is None:
            n = np.arange(v.shape[2])
        else:
            n = dims
        if id is None:
            d = np.zeros((1,))
        else:
            d = id
        if time is None:
            i = np.zeros((1, 1))
        else:
            i = np.empty((1, 1))
            i[0, 0] = time
    elif len(data.shape) == 2:
        v = np.empty((1, data.shape[0], data.shape[1]), dtype=data.dtype)
        v[0, :, :] = data
        if dims is None:
            n = np.arange(v.shape[2])
        else:
            n = dims
        if id is None:
            d = np.zeros((1,))
        else:
            d = id
        if time is None:
            i = np.empty((1, v.shape[1]))
            i[0, :] = np.arange(v.shape[1])
        else:
            i = np.empty((1, v.shape[1]), dtype=time.dtype)
            i[0, :] = time
    else:
        v = np.empty((data.shape[0], data.shape[1], data.shape[2]), dtype=data.dtype)
        v[:, :, :] = data
        if dims is None:
            n = np.arange(v.shape[2])
        else:
            n = dims
        if id is None:
            d = np.arange(data.shape[0])
        else:
            d = id
        if time is None:
            t = np.prod(v.shape[:2])
            i = np.empty((v.shape[0], v.shape[1]), dtype=int)
            i[:, :] = np.arange(t).reshape(*v.shape[:2][::-1]).T
        else:
            i = np.empty((v.shape[0], v.shape[1]), dtype=time.dtype)
            i[:, :] = time
    return v, d, i, n


def _multiple_collections(values):
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
