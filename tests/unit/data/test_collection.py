from tsfuse.data import Collection

import numpy as np
import pandas as pd

from tsfuse.data import Type, Tags
from tsfuse.data.tags import quantity, body_part
from tsfuse.transformers import Ratio


def test_create_collection_numeric():
    x = Collection(1)
    assert np.issubdtype(x.dtype, np.float64)
    assert not np.issubdtype(x.dtype, np.str_)


def test_create_collection_text():
    x = Collection('a')
    assert np.issubdtype(x.dtype, np.str_)
    assert not np.issubdtype(x.dtype, np.float64)


def test_create_collection_constant():
    x = Collection(1)
    assert x.shape == (1, 1, 1)
    assert x.type == Type.SCALAR


def test_create_collection_attributes():
    x = Collection([1, 2, 3])
    assert x.shape == (1, 1, 3)
    assert x.type == Type.ATTRIBUTES


def test_create_collection_series():
    x = Collection([[1, 2, 3], [4, 5, 6]])
    assert x.shape == (1, 2, 3)
    assert x.type == Type.SERIES


def test_create_collection_windows():
    x = Collection([[[1, 2, 3], [4, 5, 6]], [[2, 3, 4], [5, 6, 7]]])
    assert x.shape == (2, 2, 3)
    assert x.type == Type.WINDOWS


def test_create_collection_windows_variable_length():
    x = Collection([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[2, 3, 4], [5, 6, 7]]])
    assert x.shape == (2, (3, 2), 3)
    assert x.type == Type.WINDOWS


def test_tags():
    x = Collection([1, 2, 3])
    x.tags.add(quantity, 'acceleration')
    assert x.tags[quantity] == 'acceleration'
    assert x.tags['quantity'] == 'acceleration'
    assert x.tags.quantity == 'acceleration'
    x.tags.remove(quantity)
    assert x.tags[quantity] is None
    assert x.tags['quantity'] is None


def test_tag_propagation():
    x = Collection([1, 2, 3], tags=Tags({quantity: 'acceleration', body_part: 'arm_left'}))
    y = Collection([1, 2, 3], tags=Tags({quantity: 'acceleration', body_part: 'arm_right'}))
    z = Collection([1, 2, 3], tags=Tags({quantity: 'position', body_part: 'leg_right'}))
    assert quantity in Ratio().transform(x, y).tags
    assert Ratio().transform(x, y).tags.quantity == 'acceleration'
    assert body_part in Ratio().transform(x, y).tags
    assert Ratio().transform(x, y).tags.body_part == 'arm'
    assert body_part in Ratio().transform(y, z).tags
    assert Ratio().transform(y, z).tags.body_part == 'right'
    assert quantity not in Ratio().transform(x, z).tags
    assert body_part not in Ratio().transform(x, z).tags


def test_from_dataframe_time_series_fixed_length():
    df = pd.DataFrame([
        ['A', 1, 10, 20, 30],
        ['A', 2, 11, 21, 31],
        ['A', 3, 12, 22, 33],
        ['B', 1, 40, 50, 60],
        ['B', 2, 41, 51, 61],
        ['B', 3, 42, 52, 63],
    ], columns=['id', 'time', 'x', 'y', 'z'])
    collection = Collection.from_dataframe(df)
    assert collection.values.shape == (2, 3, 3)
    np.testing.assert_almost_equal(collection.values, [
        [[10, 20, 30],
         [11, 21, 31],
         [12, 22, 33],],
        [[40, 50, 60],
         [41, 51, 61],
         [42, 52, 63],],
    ])
    np.testing.assert_equal(collection.id, ['A', 'B'])


def test_to_dataframe_time_series_fixed_length():
    collection = Collection([
        [[10, 20, 30],
         [11, 21, 31],
         [12, 22, 33],],
        [[40, 50, 60],
         [41, 51, 61],
         [42, 52, 63],],
    ], index=[
        [0, 1, 2],
        [0, 1, 2],
    ], dimensions=['x', 'y', 'z'])
    df = collection.to_dataframe(column_id='id', column_sort='time')
    np.testing.assert_equal(df.columns.values, ['id', 'time', 'x', 'y', 'z'])
    np.testing.assert_equal(df.values, [
        [0, 0, 10, 20, 30],
        [0, 1, 11, 21, 31],
        [0, 2, 12, 22, 33],
        [1, 0, 40, 50, 60],
        [1, 1, 41, 51, 61],
        [1, 2, 42, 52, 63],
    ])


def test_from_dataframe_time_series_variable_length():
    df = pd.DataFrame([
        ['A', 1, 10, 20, 30],
        ['A', 2, 11, 21, 31],
        ['A', 3, 12, 22, 33],
        ['B', 1, 40, 50, 60],
        ['B', 2, 41, 51, 61],
    ], columns=['id', 'time', 'x', 'y', 'z'])
    collection = Collection.from_dataframe(df)
    np.testing.assert_almost_equal(collection.values[0].values, [
        [[10, 20, 30],
         [11, 21, 31],
         [12, 22, 33],],
    ])
    np.testing.assert_almost_equal(collection.values[1].values, [
        [[40, 50, 60],
         [41, 51, 61],],
    ])
    np.testing.assert_equal(collection.id, [['A'], ['B']])


def test_to_dataframe_time_series_variable_length():
    collection = Collection([
        [[10, 20, 30],
         [11, 21, 31],
         [12, 22, 33],],
        [[40, 50, 60],
         [41, 51, 61],],
    ], index=[
        [0, 1, 2],
        [0, 1],
    ], dimensions=['x', 'y', 'z'])
    df = collection.to_dataframe()
    np.testing.assert_equal(df.columns.values, ['id', 'time', 'x', 'y', 'z'])
    np.testing.assert_equal(df.values, [
        [0, 0, 10, 20, 30],
        [0, 1, 11, 21, 31],
        [0, 2, 12, 22, 33],
        [1, 0, 40, 50, 60],
        [1, 1, 41, 51, 61],
    ])
    

def test_from_dataframe_attributes():
    df = pd.DataFrame([
        ['A', 10, 20, 30],
        ['B', 40, 50, 60],
    ], columns=['id', 'x', 'y', 'z'])
    collection = Collection.from_dataframe(df)
    assert collection.values.shape == (2, 1, 3)
    np.testing.assert_almost_equal(collection.values, [
        [[10, 20, 30]],
        [[40, 50, 60]],
    ])
    np.testing.assert_equal(collection.id, ['A', 'B'])


def test_to_dataframe_attributes():
    collection = Collection([
        [[10, 20, 30]],
        [[40, 50, 60]],
    ], dimensions=['x', 'y', 'z'])
    df = collection.to_dataframe()
    np.testing.assert_equal(df.columns.values, ['id', 'x', 'y', 'z'])
    np.testing.assert_equal(df.values, [
        [0, 10, 20, 30],
        [1, 40, 50, 60],
    ])