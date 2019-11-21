from tsfuse.data import Collection

import numpy as np
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
