import numpy as np

from ..data import Collection

__all__ = [
    'apply',
    'apply_to_axis',
    'length',
    'transform_axis',
    'reduce_index',
    'reduce_dimensions',
    'mask_nan',
    'mask_zero',
]


def apply(calculator, *collections, **params):
    values = []
    for collection in collections:
        values.append(mask_nan(collection))
    transformed = calculator(*values, **params)
    if transformed.shape == collections[0].values.shape:
        return Collection(
            values=transformed,
            index=collections[0].index,
            dimensions=collections[0].dimensions
        )
    else:
        return Collection(transformed)


def apply_to_axis(calculator, *collections, **params):
    if 'axis' in params:
        axis = params['axis']
        del params['axis']
    else:
        axis = None
    a = transform_axis(collections[0], axis=axis)
    values = []
    for collection in collections:
        values.append(np.rollaxis(mask_nan(collection), a, start=3))
    transformed = np.rollaxis(calculator(*values, **params), 2, start=a)
    if transformed.shape == collections[0].values.shape:
        return Collection(
            values=transformed,
            index=collections[0].index,
            dimensions=collections[0].dimensions
        )
    elif transformed.shape[a] == 1:
        return Collection(
            values=transformed,
            index=reduce_index(collections[0], a),
            dimensions=reduce_dimensions(collections[0], a)
        )
    else:
        return Collection(transformed)


def length(collection, axis):
    a = transform_axis(collection, axis=axis)
    return collection.shape[a]


def transform_axis(collection, axis=None):
    if isinstance(axis, int) and (axis >= 0) and (axis <= 2):
        return axis
    if axis is None:
        axis = collection.transform_axis
    if axis == 'dimensions':
        return 2
    elif axis == 'timestamps':
        return 1
    elif axis == 'windows':
        return 0


def reduce_index(collection, a):
    if a == 2:
        return collection.index
    elif a == 1:
        return collection.index[:, :1]
    elif a == 0:
        return collection.index[:1, :]


def reduce_dimensions(collection, a):
    if a == 2:
        return collection.dimensions[:1]
    elif a == 1:
        return collection.dimensions
    elif a == 0:
        return collection.dimensions


def mask_nan(collection):
    """
    Replace the mask value by ``np.nan`` for all values in a data collection.

    Parameters
    ----------
    collection : NumPyDataCollection
        NumPy data collection.

    Returns
    -------
    replaced : np.ndarray
        Values array where each occurrence of ``collection.mask_value`` is
        replaced by ``np.nan``.
    """
    if not np.isnan(collection.mask_value):
        array = np.array(collection.values, copy=True)
        array[array == collection.mask_value] = np.nan
        return array
    else:
        return collection.values


def mask_zero(collection):
    """
    Replace the mask value by zero for all values in a data collection.

    Parameters
    ----------
    collection : NumPyDataCollection
        NumPy data collection.

    Returns
    -------
    replaced : np.ndarray
        Values array where each occurrence of ``collection.mask_value`` is
        replaced by zero.
    """
    array = np.array(collection.values, copy=True)
    if not np.isnan(collection.mask_value):
        array[array == collection.mask_value] = 0
    else:
        array[np.isnan(array)] = 0
    return array
