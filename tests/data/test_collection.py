from tsfuse.data import Collection

import numpy as np
import pandas as pd

from tsfuse.data import Type, Tags
from tsfuse.data.tags import quantity, body_part
from tsfuse.transformers import Ratio


def test_create_collection_attributes():
    df = pd.DataFrame({
        'id': [0, 1, 2],
        'x1': [1, 2, 3],
        'x2': [2, 3, 4],
    })
    x = Collection(df)
    assert x.shape == (3, 1, 2)
    assert x.type == Type.WINDOWS
    np.testing.assert_equal(x.id, [0, 1, 2])
    np.testing.assert_equal(x.dims, ['x1', 'x2'])


def test_create_collection_series_fixed_length():
    df = pd.DataFrame({
        'id': [0, 0, 0, 1, 1, 1],
        'x1': [1, 2, 3, 1, 2, 3],
        'x2': [2, 2, 2, 4, 4, 4],
        'time': [0, 1, 2, 0, 1, 2],
    })
    x = Collection(df)
    assert x.shape == (2, 3, 2)
    assert x.type == Type.WINDOWS
    np.testing.assert_equal(x.id, [0, 1])
    np.testing.assert_equal(x.time, [[0, 1, 2], [0, 1, 2]])
    np.testing.assert_equal(x.dims, ['x1', 'x2'])


def test_create_collection_series_variable_length():
    df = pd.DataFrame({
        'id': [0, 0, 0, 1, 1],
        'x1': [1, 2, 3, 1, 2],
        'x2': [2, 2, 2, 4, 4],
        'time': [0, 1, 2, 0, 1],
    })
    x = Collection(df)
    assert x.shape == (2, (3, 2), 2)
    assert x.type == Type.WINDOWS
    np.testing.assert_equal(x.id, [0, 1])
    np.testing.assert_equal(x.time, [[0, 1, 2], [0, 1]])
    np.testing.assert_equal(x.dims, ['x1', 'x2'])


def test_to_dataframe_attributes():
    collection = Collection.from_array([
        [[10, 20, 30]],
        [[40, 50, 60]],
    ], dims=['x', 'y', 'z'])
    print(collection.id)
    df = collection.to_dataframe()
    np.testing.assert_equal(df.columns.values, ['id', 'x', 'y', 'z'])
    np.testing.assert_equal(df.values, [
        [0, 10, 20, 30],
        [1, 40, 50, 60],
    ])


def test_to_dataframe_time_series_fixed_length():
    collection = Collection.from_array([
        [[10, 20, 30],
         [11, 21, 31],
         [12, 22, 33],],
        [[40, 50, 60],
         [41, 51, 61],
         [42, 52, 63],],
    ], time=[
        [0, 1, 2],
        [0, 1, 2],
    ], dims=['x', 'y', 'z'])
    df = collection.to_dataframe()
    np.testing.assert_equal(df.columns.values, ['id', 'time', 'x', 'y', 'z'])
    np.testing.assert_equal(df.values, [
        [0, 0, 10, 20, 30],
        [0, 1, 11, 21, 31],
        [0, 2, 12, 22, 33],
        [1, 0, 40, 50, 60],
        [1, 1, 41, 51, 61],
        [1, 2, 42, 52, 63],
    ])


def test_to_dataframe_time_series_variable_length():
    collection = Collection.from_array([
        [[10, 20, 30],
         [11, 21, 31],
         [12, 22, 33],],
        [[40, 50, 60],
         [41, 51, 61],],
    ], time=[
        [0, 1, 2],
        [0, 1],
    ], dims=['x', 'y', 'z'])
    df = collection.to_dataframe()
    np.testing.assert_equal(df.columns.values, ['id', 'time', 'x', 'y', 'z'])
    np.testing.assert_equal(df.values, [
        [0, 0, 10, 20, 30],
        [0, 1, 11, 21, 31],
        [0, 2, 12, 22, 33],
        [1, 0, 40, 50, 60],
        [1, 1, 41, 51, 61],
    ])


def test_create_collection_numeric():
    x = Collection.from_array(1)
    assert np.issubdtype(x.dtype, np.float64)
    assert not np.issubdtype(x.dtype, np.str_)


def test_create_collection_text():
    x = Collection.from_array('a')
    assert np.issubdtype(x.dtype, np.str_)
    assert not np.issubdtype(x.dtype, np.float64)


def test_create_collection_constant():
    x = Collection.from_array(1)
    assert x.shape == (1, 1, 1)
    assert x.type == Type.SCALAR


def test_create_collection_attributes():
    x = Collection.from_array([1, 2, 3])
    assert x.shape == (1, 1, 3)
    assert x.type == Type.ATTRIBUTES


def test_create_collection_series():
    x = Collection.from_array([[1, 2, 3], [4, 5, 6]])
    assert x.shape == (1, 2, 3)
    assert x.type == Type.SERIES


def test_create_collection_windows():
    x = Collection.from_array([[[1, 2, 3], [4, 5, 6]], [[2, 3, 4], [5, 6, 7]]])
    assert x.shape == (2, 2, 3)
    assert x.type == Type.WINDOWS


def test_create_collection_windows_variable_length():
    x = Collection.from_array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[2, 3, 4], [5, 6, 7]]])
    assert x.shape == (2, (3, 2), 3)
    assert x.type == Type.WINDOWS


def test_tags():
    x = Collection.from_array([1, 2, 3])
    x.tags.add(quantity, 'acceleration')
    assert x.tags[quantity] == 'acceleration'
    assert x.tags['quantity'] == 'acceleration'
    assert x.tags.quantity == 'acceleration'
    x.tags.remove(quantity)
    assert x.tags[quantity] is None
    assert x.tags['quantity'] is None


def test_tag_propagation():
    x = Collection.from_array([1, 2, 3], tags=Tags({quantity: 'acceleration', body_part: 'arm_left'}))
    y = Collection.from_array([1, 2, 3], tags=Tags({quantity: 'acceleration', body_part: 'arm_right'}))
    z = Collection.from_array([1, 2, 3], tags=Tags({quantity: 'position', body_part: 'leg_right'}))
    assert quantity in Ratio().transform(x, y).tags
    assert Ratio().transform(x, y).tags.quantity == 'acceleration'
    assert body_part in Ratio().transform(x, y).tags
    assert Ratio().transform(x, y).tags.body_part == 'arm'
    assert body_part in Ratio().transform(y, z).tags
    assert Ratio().transform(y, z).tags.body_part == 'right'
    assert quantity not in Ratio().transform(x, z).tags
    assert body_part not in Ratio().transform(x, z).tags
