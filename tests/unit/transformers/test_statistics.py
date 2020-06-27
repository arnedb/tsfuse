import pytest
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.tsa.ar_model

from tsfuse.data.synthetic import series, brownian
from tsfuse.data import Collection
from tsfuse.transformers.statistics import *


@pytest.fixture
def x():
    return brownian()


@pytest.fixture
def y():
    return brownian()


def test_single_pass_statistics(x):
    result = SinglePassStatistics().transform(x)
    for i, a in series(x):
        actual = result.values[i]
        np.testing.assert_almost_equal(actual[0], len(a))
        np.testing.assert_almost_equal(actual[1], np.sum(a))
        np.testing.assert_almost_equal(actual[2], np.min(a))
        np.testing.assert_almost_equal(actual[3], np.max(a))
        np.testing.assert_almost_equal(actual[4], np.mean(a))
        np.testing.assert_almost_equal(actual[5], np.var(a))
        np.testing.assert_almost_equal(actual[6], stats.skew(a))
        np.testing.assert_almost_equal(actual[7], stats.kurtosis(a))


def test_sum(x):
    result = Sum().transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.sum(a)
        np.testing.assert_almost_equal(actual, expected)


def test_min():
    x = Collection([4, 3, 2, 1, 2, 3, 4])
    actual = Min().transform(x).values
    np.testing.assert_equal(actual, 1)


def test_arg_min():
    x = Collection([4, 3, 2, 1, 2, 3, 4])
    actual = ArgMin().transform(x).values
    np.testing.assert_equal(actual, 3)


def test_max():
    x = Collection([1, 2, 3, 4, 3, 2, 1])
    actual = Max().transform(x).values
    np.testing.assert_equal(actual, 4)


def test_arg_max():
    x = Collection([1, 2, 3, 4, 3, 2, 1])
    actual = ArgMax().transform(x).values
    np.testing.assert_equal(actual, 3)


def test_mean(x):
    result = Mean().transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.mean(a)
        np.testing.assert_almost_equal(actual, expected)


def test_median(x):
    result = Median().transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.median(a)
        np.testing.assert_almost_equal(actual, expected)


def test_variance(x):
    result = Variance().transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.var(a)
        np.testing.assert_almost_equal(actual, expected)


def test_skewness(x):
    result = Skewness().transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = stats.skew(a)
        np.testing.assert_almost_equal(actual, expected)


def test_kurtosis(x):
    result = Kurtosis().transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = stats.kurtosis(a)
        np.testing.assert_almost_equal(actual, expected)


def test_quantile_zero(x):
    result = Quantile(q=0).transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.percentile(a, q=0)
        np.testing.assert_almost_equal(actual, expected)


def test_quantile_half(x):
    result = Quantile(q=0.5).transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.percentile(a, q=50)
        np.testing.assert_almost_equal(actual, expected)


def test_quantile_one(x):
    result = Quantile(q=1).transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.percentile(a, q=100)
        np.testing.assert_almost_equal(actual, expected)


def test_index_mass_quantile_absolute(x):
    result = IndexMassQuantile(q=0.5, rel=False).transform(x)
    for i, a in series(x):
        actual = result.values[i]
        a = np.abs(a)
        expected = np.argmax((np.cumsum(a) / np.sum(a)) >= 0.5) + 1
        np.testing.assert_almost_equal(actual, expected)


def test_index_mass_quantile_relative(x):
    result = IndexMassQuantile(q=0.5, rel=True).transform(x)
    for i, a in series(x):
        actual = result.values[i]
        a = np.abs(a)
        expected = (np.argmax((np.cumsum(a) / np.sum(a)) >= 0.5) + 1) / len(a)
        np.testing.assert_almost_equal(actual, expected)


def test_energy(x):
    result = Energy().transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.sum(np.square(a))
        np.testing.assert_almost_equal(actual, expected)


def test_energy_ratio():
    x = Collection([1, 2, 3, 4, 5, 6, 7, 8])
    chunks = [[1, 2, 3], [4, 5, 6], [7, 8]]
    result = EnergyRatio(chunks=3).transform(x).values
    assert result.shape == (1, 1, 3)
    total = np.sum(np.square([1, 2, 3, 4, 5, 6, 7, 8]))
    np.testing.assert_almost_equal(result[0, 0, 0], np.sum(np.square(chunks[0])) / total)
    np.testing.assert_almost_equal(result[0, 0, 1], np.sum(np.square(chunks[1])) / total)
    np.testing.assert_almost_equal(result[0, 0, 2], np.sum(np.square(chunks[2])) / total)


def test_entropy(x):
    x = Collection(np.abs(x.values) + 1)  # ensure that values > 0
    result = Entropy().transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = stats.entropy(a)
        np.testing.assert_almost_equal(actual, expected)

    
def test_sample_entropy(x):
    x = Collection([1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5])
    result = SampleEntropy().transform(x)
    actual = result.values
    expected = 0.55961579
    # Computed using Physionet sample entropy implementation:
    # https://physionet.org/content/sampen/1.0.0/
    np.testing.assert_almost_equal(actual, expected)


def test_binned_distribution(x):
    result = BinnedDistribution(bins=10).transform(x)
    for i, a in series(x):
        actual = result.values[i]
        r = (np.min(a), np.max(a))
        s = (r[1] - r[0]) / 10
        for bin in range(10):
            lower = r[0] + bin * s
            upper = r[0] + (bin + 1) * s
            if bin < 10 - 1:
                count = len(np.where((a >= lower) & (a < upper))[0])
            else:
                count = len(np.where(a >= lower)[0])
            ratio = count / len(a)
            np.testing.assert_almost_equal(actual[bin], ratio)


def test_binned_entropy(x):
    result = BinnedEntropy(bins=10).transform(x)
    for i, a in series(x):
        actual = result.values[i]
        r = (np.min(a), np.max(a))
        s = (r[1] - r[0]) / 10
        bd = []
        for bin in range(10):
            lower = r[0] + bin * s
            upper = r[0] + (bin + 1) * s
            if bin < 10 - 1:
                count = len(np.where((a >= lower) & (a < upper))[0])
            else:
                count = len(np.where(a >= lower)[0])
            bd.append(count / len(a))
        np.testing.assert_almost_equal(actual, stats.entropy(bd))


def test_c3(x):
    result = C3(lag=1).transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.mean(np.square(a[2:]) * a[1:-1] * a[:-2])
        np.testing.assert_almost_equal(actual, expected)


def test_cid(x):
    result = CID().transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.sqrt(np.sum(np.square(np.diff(a))))
        np.testing.assert_almost_equal(actual, expected)


def test_count_above_mean(x):
    result = CountAboveMean().transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.sum(a > np.mean(a))
        np.testing.assert_almost_equal(actual, expected)


def test_count_below_mean(x):
    result = CountBelowMean().transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.sum(a < np.mean(a))
        np.testing.assert_almost_equal(actual, expected)


def test_range_count(x):
    result = RangeCount(min=0, max=1).transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.sum((a >= 0) & (a < 1))
        np.testing.assert_almost_equal(actual, expected)


def test_value_count():
    x = Collection([1, 1, 2, 3])
    actual = ValueCount(value=1).transform(x).values
    np.testing.assert_equal(actual, 2)


def test_outliers(x):
    result = Outliers(r=1.5).transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.sum(np.abs(a - np.mean(a)) > 1.5 * np.std(a))
        np.testing.assert_almost_equal(actual, expected)


def test_outliers_rel(x):
    result = Outliers(r=1.5, rel=True).transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.sum(np.abs(a - np.mean(a)) > 1.5 * np.std(a)) / len(a)
        np.testing.assert_almost_equal(actual, expected)


def test_cross_correlation(x, y):
    result = CrossCorrelation().transform(x, y)
    for i, a, b in series(x, y):
        actual = result.values[i]
        np.testing.assert_almost_equal(actual[0], np.correlate(a, b)[0])
        for lag in range(1, len(actual)):
            np.testing.assert_almost_equal(actual[lag], np.correlate(a[lag:], b[:-lag])[0])


def test_auto_correlation(x):
    result = AutoCorrelation().transform(x)
    for i, a in series(x):
        actual = result.values[i]
        mu = np.mean(a)
        var = np.var(a)
        np.testing.assert_almost_equal(actual[0], np.mean((a - mu) * (a - mu)) / var)
        for lag in range(1, len(actual)):
            expected = np.mean((a[:-lag] - mu) * (a[lag:] - mu)) / var
            np.testing.assert_almost_equal(actual[lag], expected)


def test_auto_regressive_coefficients(x):
    result = AutoRegressiveCoefficients().transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = statsmodels.tsa.ar_model.AR(a).fit().params
        np.testing.assert_almost_equal(actual, expected)


def test_high_variance_0(x):
    result = HighVariance(threshold=0).transform(x)
    assert np.issubdtype(result.dtype, np.bool_)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.var(a) > 0
        np.testing.assert_almost_equal(actual, expected)


def test_high_variance_1(x):
    result = HighVariance(threshold=1).transform(x)
    assert np.issubdtype(result.dtype, np.bool_)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.var(a) > 1
        np.testing.assert_almost_equal(actual, expected)


def test_high_standard_deviation(x):
    result = HighStandardDeviation(r=0.1).transform(x)
    assert np.issubdtype(result.dtype, np.bool_)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.std(a) > 0.1 * (np.max(a) - np.min(a))
        np.testing.assert_almost_equal(actual, expected)


def test_high_standard_deviation_zero(x):
    result = HighStandardDeviation(r=0).transform(x)
    assert np.issubdtype(result.dtype, np.bool_)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.std(a) > 0 * (np.max(a) - np.min(a))
        np.testing.assert_almost_equal(actual, expected)


def test_high_standard_deviation_one(x):
    result = HighStandardDeviation(r=1).transform(x)
    assert np.issubdtype(result.dtype, np.bool_)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.std(a) > 1 * (np.max(a) - np.min(a))
        np.testing.assert_almost_equal(actual, expected)


def test_symmetry_looking_0(x):
    result = SymmetryLooking(r=0).transform(x)
    assert np.issubdtype(result.dtype, np.bool_)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.abs(np.mean(a) - np.median(a)) < 0 * (np.max(a) - np.min(a))
        np.testing.assert_almost_equal(actual, expected)


def test_symmetry_looking_1(x):
    result = SymmetryLooking(r=1).transform(x)
    assert np.issubdtype(result.dtype, np.bool_)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.abs(np.mean(a) - np.median(a)) < 1 * (np.max(a) - np.min(a))
        np.testing.assert_almost_equal(actual, expected)


def test_number_crossings_zero_no():
    x = Collection([1, 2, 3, 4])
    actual = NumberCrossings().transform(x).values
    np.testing.assert_equal(actual, 0)


def test_number_crossings_zero_one():
    x = Collection([-1, 1, 2, 3, 4])
    actual = NumberCrossings().transform(x).values
    np.testing.assert_equal(actual, 1)


def test_number_crossings_zero_two():
    x = Collection([1, -1, 2, 3, 4])
    actual = NumberCrossings().transform(x).values
    np.testing.assert_equal(actual, 2)


def test_number_crossings_one():
    x = Collection([0, 2, 3, 4])
    actual = NumberCrossings().transform(x).values
    np.testing.assert_equal(actual, 1)


def test_linear_trend(x):
    result = LinearTrend().transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = stats.linregress(np.arange(len(a)), a)
        np.testing.assert_almost_equal(actual, expected)


def test_longest_strike_above_mean_zero():
    x = Collection([1, 1, 1])
    actual = LongestStrikeAboveMean().transform(x).values
    np.testing.assert_equal(actual, 0)


def test_longest_strike_above_mean_two():
    x = Collection([0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 0])
    actual = LongestStrikeAboveMean().transform(x).values
    np.testing.assert_equal(actual, 2)


def test_longest_strike_above_mean_three():
    x = Collection([0, 0, 0, 2, 2, 0, 0, 2, 2, 2, 0, 0, 2, 0])
    actual = LongestStrikeAboveMean().transform(x).values
    np.testing.assert_equal(actual, 3)


def test_sum_change(x):
    result = SumChange().transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.sum(np.diff(a))
        np.testing.assert_almost_equal(actual, expected)


def test_sum_change_abs(x):
    result = SumChange(abs=True).transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.sum(np.abs(np.diff(a)))
        np.testing.assert_almost_equal(actual, expected)


def test_mean_change(x):
    result = MeanChange().transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.mean(np.diff(a))
        np.testing.assert_almost_equal(actual, expected)


def test_mean_change_abs(x):
    result = MeanChange(abs=True).transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.mean(np.abs(np.diff(a)))
        np.testing.assert_almost_equal(actual, expected)


def test_mean_second_derivative_central(x):
    result = MeanSecondDerivativeCentral().transform(x)
    for i, a in series(x):
        actual = result.values[i]
        expected = np.mean(a[2:] - 2 * a[1:-1] + a[:-2])
        np.testing.assert_almost_equal(actual, expected)


def test_time_reversal_asymmetry_statistic(x):
    result = TimeReversalAsymmetryStatistic(lag=1).transform(x)
    for i, a in series(x):
        actual = result.values[i]
        n = len(a) - 2
        expected = np.mean(a[2:][:n] * a[2:][:n] * a[1:][:n] - a[1:][:n] * a[:n] * a[:n])
        np.testing.assert_almost_equal(actual, expected)


def test_friedrich_coefficients(x):
    result = FriedrichCoefficients(m=1, r=10).transform(x)
    for i, a in series(x):
        actual = result.values[i]
        df = pd.DataFrame({'signal': a[:-1], 'delta': np.diff(a)})
        df['quantiles'] = pd.qcut(df.signal, 10)
        q = df.groupby('quantiles')
        mean = pd.DataFrame({'x': q.signal.mean(), 'y': q.delta.mean()})
        expected = np.polyfit(mean.x, mean.y, deg=1)
        np.testing.assert_almost_equal(actual, expected)


def test_friedrich_coefficients_m_3(x):
    result = FriedrichCoefficients(m=3, r=10).transform(x)
    for i, a in series(x):
        actual = result.values[i]
        df = pd.DataFrame({'signal': a[:-1], 'delta': np.diff(a)})
        df['quantiles'] = pd.qcut(df.signal, 10)
        q = df.groupby('quantiles')
        mean = pd.DataFrame({'x': q.signal.mean(), 'y': q.delta.mean()})
        expected = np.polyfit(mean.x, mean.y, deg=3)
        np.testing.assert_almost_equal(actual, expected)


def test_max_langevin_fixed_point(x):
    result = MaxLangevinFixedPoint(m=1, r=10).transform(x)
    for i, a in series(x):
        actual = result.values[i]
        df = pd.DataFrame({'signal': a[:-1], 'delta': np.diff(a)})
        df['quantiles'] = pd.qcut(df.signal, 10)
        q = df.groupby('quantiles')
        mean = pd.DataFrame({'x': q.signal.mean(), 'y': q.delta.mean()})
        coef = np.polyfit(mean.x, mean.y, deg=1)
        expected = np.max(np.roots(coef).real, keepdims=True)
        np.testing.assert_almost_equal(actual, expected)


def test_max_langevin_fixed_point_m_3(x):
    result = MaxLangevinFixedPoint(m=3, r=10).transform(x)
    for i, a in series(x):
        actual = result.values[i]
        df = pd.DataFrame({'signal': a[:-1], 'delta': np.diff(a)})
        df['quantiles'] = pd.qcut(df.signal, 10)
        q = df.groupby('quantiles')
        mean = pd.DataFrame({'x': q.signal.mean(), 'y': q.delta.mean()})
        coef = np.polyfit(mean.x, mean.y, deg=3)
        expected = np.max(np.roots(coef).real)
        np.testing.assert_almost_equal(actual, expected)
