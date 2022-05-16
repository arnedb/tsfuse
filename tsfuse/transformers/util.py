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
        return Collection.from_array(
            transformed,
            time=collections[0].time,
            dims=collections[0].dims
        )
    else:
        return Collection.from_array(transformed)


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
        return Collection.from_array(
            transformed,
            time=collections[0].time,
            dims=collections[0].dims
        )
    elif transformed.shape[a] == 1:
        return Collection.from_array(
            transformed,
            time=reduce_index(collections[0], a),
            dims=reduce_dimensions(collections[0], a)
        )
    else:
        return Collection.from_array(transformed)


def length(collection, axis):
    a = transform_axis(collection, axis=axis)
    return collection.shape[a]


def transform_axis(collection, axis=None):
    if isinstance(axis, int) and (axis >= 0) and (axis <= 2):
        return axis
    if axis is None:
        axis = collection.transform_axis
    if axis == 'dims':
        return 2
    elif axis == 'time':
        return 1
    elif axis == 'windows':
        return 0


def reduce_index(collection, a):
    if a == 2:
        return collection.time
    elif a == 1:
        return collection.time[:, :1]
    elif a == 0:
        return collection.time[:1, :]


def reduce_dimensions(collection, a):
    if a == 2:
        return collection.dims[:1]
    elif a == 1:
        return collection.dims
    elif a == 0:
        return collection.dims


def mask_nan(collection):
    return collection.values


def mask_zero(collection):
    array = np.array(collection.values, copy=True)
    array[np.isnan(array)] = 0
    return array
