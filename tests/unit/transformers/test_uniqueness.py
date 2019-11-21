import numpy as np

from tsfuse.transformers.uniqueness import *
from tsfuse.data import Collection


def test_has_duplicate_true():
    x = Collection([1, 2, 3, 3])
    actual = HasDuplicate().transform(x).values
    np.testing.assert_equal(actual, True)


def test_has_duplicate_false():
    x = Collection([1, 2, 3, 4])
    actual = HasDuplicate().transform(x).values
    np.testing.assert_equal(actual, False)


def test_has_duplicate_min_true():
    x = Collection([1, 1, 2, 3])
    actual = HasDuplicateMin().transform(x).values
    np.testing.assert_equal(actual, True)


def test_has_duplicate_min_false():
    x = Collection([2, 3, 4, 4])
    actual = HasDuplicateMin().transform(x).values
    np.testing.assert_equal(actual, False)


def test_has_duplicate_max_true():
    x = Collection([2, 3, 4, 4])
    actual = HasDuplicateMax().transform(x).values
    np.testing.assert_equal(actual, True)


def test_has_duplicate_max_false():
    x = Collection([1, 1, 2, 3])
    actual = HasDuplicateMax().transform(x).values
    np.testing.assert_equal(actual, False)


def test_has_duplicate_empty():
    x = Collection([np.nan])
    actual = HasDuplicate().transform(x).values
    np.testing.assert_equal(actual, False)


def test_number_of_unique_values_rel():
    x = Collection([1, 2, 3, 4])
    actual = NumberUniqueValues(rel=True).transform(x).values
    np.testing.assert_equal(actual, 1.0)


def test_number_of_unique_values_abs():
    x = Collection([1, 2, 3, 4])
    actual = NumberUniqueValues(rel=False).transform(x).values
    np.testing.assert_equal(actual, 4)


def test_number_of_unique_values_1_rel():
    x = Collection([2, 2, 2, 2])
    actual = NumberUniqueValues(rel=True).transform(x).values
    np.testing.assert_equal(actual, 0.25)


def test_number_of_unique_values_1_abs():
    x = Collection([2, 2, 2, 2])
    actual = NumberUniqueValues(rel=False).transform(x).values
    np.testing.assert_equal(actual, 1)


def test_number_of_unique_values_0_rel():
    x = Collection([np.nan])
    actual = NumberUniqueValues(rel=True).transform(x).values
    np.testing.assert_equal(actual, np.nan)


def test_number_of_unique_values_0_abs():
    x = Collection([np.nan])
    actual = NumberUniqueValues(rel=False).transform(x).values
    np.testing.assert_equal(actual, np.nan)


def test_sum_of_reoccurring_data_poins():
    x = Collection([1, 1, 2, 3, 3, 4])
    actual = SumReoccurringDataPoints().transform(x).values
    np.testing.assert_equal(actual, 8)


def test_sum_of_reoccurring_data_points_0():
    x = Collection([1, 2, 3, 4])
    actual = SumReoccurringDataPoints().transform(x).values
    np.testing.assert_equal(actual, 0)


def test_sum_of_reoccurring_values():
    x = Collection([1, 1, 2, 3, 3, 4])
    actual = SumReoccurringValues().transform(x).values
    np.testing.assert_equal(actual, 4)


def test_sum_of_reoccurring_values_0():
    x = Collection([1, 2, 3, 4])
    actual = SumReoccurringValues().transform(x).values
    np.testing.assert_equal(actual, 0)
