import numpy as np
from scipy import stats
import statsmodels.tsa.ar_model
from statsmodels.tools.sm_exceptions import MissingDataError

from ..computation import Transformer, Graph, Constant
from .util import apply_to_axis, length
from .calculators.statistics import *
from .queries import Count, Slice
from .mathematics import Abs, Diff, Square, Sqrt, Roots, Exponent, Sum, CumSum
from .boolean import Equal

__all__ = [
    "Length",
    "Sum",
    "Mean",
    "Median",
    "Min",
    "ArgMin",
    "Max",
    "ArgMax",
    "Variance",
    "StandardDeviation",
    "Skewness",
    "Kurtosis",
    "SpectralMoment",
    "SpectralMean",
    "SpectralVariance",
    "SpectralSkewness",
    "SpectralKurtosis",
    "SinglePassStatistics",
    "Quantile",
    "IndexMassQuantile",
    "Energy",
    "EnergyRatio",
    "Entropy",
    "SampleEntropy",
    "BinnedDistribution",
    "BinnedEntropy",
    "C3",
    "CID",
    "CountAboveMean",
    "CountBelowMean",
    "RangeCount",
    "ValueCount",
    "Outliers",
    "AutoCorrelation",
    "CrossCorrelation",
    "AutoRegressiveCoefficients",
    "HighVariance",
    "HighStandardDeviation",
    "SymmetryLooking",
    "NumberCrossings",
    "LinearTrend",
    "LongestStrikeAboveMean",
    "LongestStrikeBelowMean",
    "SumChange",
    "MeanChange",
    "MeanSecondDerivativeCentral",
    "TimeReversalAsymmetryStatistic",
    "FriedrichCoefficients",
    "MaxLangevinFixedPoint",
]


class SinglePassStatistics(Transformer):
    """
    Eight simple statistics

    Calculates the following statistics using only one pass over the data:

    - length
    - sum
    - min
    - max
    - mean
    - variance
    - skewness
    - kurtosis
    """

    def __init__(self, *parents, axis=None, **kwargs):
        super(SinglePassStatistics, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def apply(self, x):
        return apply_to_axis(single_pass_statistics, x, axis=self.axis)


class Length(Transformer):
    """
    Number of samples

    Parameters
    ----------
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, axis=None, **kwargs):
        super(Length, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
        ]

    def apply(self, x):
        def calculator(a):
            return np.sum(~np.isnan(a), axis=-1, keepdims=True)

        return apply_to_axis(calculator, x, axis=self.axis)


class Mean(Transformer):
    """
    Arithmetic mean

    Parameters
    ----------
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, axis=None, **kwargs):
        super(Mean, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def graph(self, x):
        return Graph(
            Slice(SinglePassStatistics(x, axis=self.axis), i=4, axis=self.axis)
        )


class Median(Transformer):
    """
    Middle value of the sorted values

    Parameters
    ----------
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, axis=None, **kwargs):
        super(Median, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def apply(self, x):
        def calculator(a):
            return np.nanmedian(a, keepdims=True, axis=-1)

        return apply_to_axis(calculator, x, axis=self.axis)


class Min(Transformer):
    """
    Smallest value

    Parameters
    ----------
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, axis=None, **kwargs):
        super(Min, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def graph(self, x):
        return Graph(
            Slice(SinglePassStatistics(x, axis=self.axis), i=2, axis=self.axis)
        )


class ArgMin(Transformer):
    """
    Location of smallest value

    Parameters
    ----------
    first : bool, default: True
        If true, return first location. Otherwise, return last location.
    rel : bool, default: False
        Return location relative to total length (i.e., as a number in [0, 1])
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, first=True, rel=False, axis=None, **kwargs):
        super(ArgMin, self).__init__(*parents, **kwargs)
        self.first = first
        self.rel = rel
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64)
            or np.issubdtype(x.dtype, np.bool_),
        ]

    def apply(self, x):
        def calculator(a):
            if self.first:
                values = np.expand_dims(np.nanargmin(a, axis=-1), axis=-1)
            else:
                values = np.expand_dims(
                    np.nanargmin(np.flip(a, axis=-1), axis=-1), axis=-1
                )
                values = -values + a.shape[-1] - 1
            if self.rel:
                n = np.nansum(~np.isnan(a), axis=-1, keepdims=True)
                values = values / n
            return values

        return apply_to_axis(calculator, x, axis=self.axis)


class Max(Transformer):
    """
    Largest value

    Parameters
    ----------
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, axis=None, **kwargs):
        super(Max, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def graph(self, x):
        return Graph(
            Slice(SinglePassStatistics(x, axis=self.axis), i=3, axis=self.axis)
        )


class ArgMax(Transformer):
    """
    Location of largest value

    Parameters
    ----------
    first : bool, default: True
        If true, return first location. Otherwise, return last location.
    rel : bool, default: False
        Return location relative to total length (i.e., as a number in [0, 1])
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, first=True, rel=False, axis=None, **kwargs):
        super(ArgMax, self).__init__(*parents, **kwargs)
        self.first = first
        self.rel = rel
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64)
            or np.issubdtype(x.dtype, np.bool_),
        ]

    def apply(self, x):
        def calculator(a):
            if self.first:
                values = np.expand_dims(np.nanargmax(a, axis=-1), axis=-1)
            else:
                values = np.expand_dims(
                    np.nanargmax(np.flip(a, axis=-1), axis=-1), axis=-1
                )
                values = -values + a.shape[-1] - 1
            if self.rel:
                n = np.nansum(~np.isnan(a), axis=-1, keepdims=True)
                values = values / n
            return values

        return apply_to_axis(calculator, x, axis=self.axis)


class Variance(Transformer):
    """
    Variance

    Parameters
    ----------
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, axis=None, **kwargs):
        super(Variance, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def graph(self, x):
        return Graph(
            Slice(SinglePassStatistics(x, axis=self.axis), i=5, axis=self.axis)
        )


class StandardDeviation(Transformer):
    """
    Standard deviation

    Parameters
    ----------
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, axis=None, **kwargs):
        super(StandardDeviation, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def graph(self, x):
        return Graph(Sqrt(Variance(x, axis=self.axis)))


class Skewness(Transformer):
    """
    Skewness

    Parameters
    ----------
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, axis=None, **kwargs):
        super(Skewness, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def graph(self, x):
        return Graph(
            Slice(SinglePassStatistics(x, axis=self.axis), i=6, axis=self.axis)
        )


class Kurtosis(Transformer):
    """
    Kurtosis

    Parameters
    ----------
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, axis=None, **kwargs):
        super(Kurtosis, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def graph(self, x):
        return Graph(
            Slice(SinglePassStatistics(x, axis=self.axis), i=7, axis=self.axis)
        )


class SpectralMoment(Transformer):
    """
    Spectral moment

    Parameters
    ----------
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, r=1, origin=False, axis=None, **kwargs):
        super(SpectralMoment, self).__init__(*parents, **kwargs)
        self.r = r
        self.origin = origin
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def apply(self, x):
        return apply_to_axis(
            spectral_moment, x, r=self.r, origin=self.origin, axis=self.axis
        )


class SpectralMean(Transformer):
    """
    Spectral mean

    Parameters
    ----------
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, origin=False, axis=None, **kwargs):
        super(SpectralMean, self).__init__(*parents, **kwargs)
        self.origin = origin
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def graph(self, x):
        return Graph(SpectralMoment(x, r=1, origin=self.origin, axis=self.axis))


class SpectralVariance(Transformer):
    """
    Spectral variance

    Parameters
    ----------
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, origin=False, axis=None, **kwargs):
        super(SpectralVariance, self).__init__(*parents, **kwargs)
        self.origin = origin
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def graph(self, x):
        m2 = SpectralMoment(x, axis=self.axis, r=2, origin=self.origin)
        mean = SpectralMean(x, axis=self.axis, origin=self.origin)
        return Graph(m2 - Exponent(mean, a=2))


class SpectralSkewness(Transformer):
    """
    Spectral skewness

    Parameters
    ----------
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, origin=False, axis=None, **kwargs):
        super(SpectralSkewness, self).__init__(*parents, **kwargs)
        self.origin = origin
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def graph(self, x):
        m3 = SpectralMoment(x, axis=self.axis, r=3, origin=self.origin)
        mean = SpectralMean(x, axis=self.axis, origin=self.origin)
        variance = SpectralVariance(x, axis=self.axis, origin=self.origin)
        return Graph(
            (m3 - mean * variance * Constant(3) - Exponent(mean, a=3))
            / Exponent(variance, a=1.5)
        )


class SpectralKurtosis(Transformer):
    """
    Spectral kurtosis

    Parameters
    ----------
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, origin=False, axis=None, **kwargs):
        super(SpectralKurtosis, self).__init__(*parents, **kwargs)
        self.origin = origin
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def graph(self, x):
        m2 = SpectralMoment(x, axis=self.axis, r=2, origin=self.origin)
        m3 = SpectralMoment(x, axis=self.axis, r=3, origin=self.origin)
        m4 = SpectralMoment(x, axis=self.axis, r=4, origin=self.origin)
        mean = SpectralMean(x, axis=self.axis, origin=self.origin)
        variance = SpectralVariance(x, axis=self.axis, origin=self.origin)
        return Graph(
            (
                m4
                - mean * m3 * Constant(4)
                + m2 * Exponent(mean, a=2) * Constant(6)
                - mean * Constant(3)
            )
            / Exponent(variance, a=2)
        )


class Quantile(Transformer):
    """
    q-Quantile

    Parameters
    ----------
    q : float, default: 0.5
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, q=0.5, axis=None, **kwargs):
        super(Quantile, self).__init__(*parents, **kwargs)
        self.q = q
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def apply(self, x):
        def calculator(a):
            return np.nanpercentile(a, q=(self.q * 100), keepdims=True, axis=-1)

        return apply_to_axis(calculator, x, axis=self.axis)


class IndexMassQuantile(Transformer):
    """
    Index mass q-quantile

    Parameters
    ----------
    q : float, default: 0.5
    rel : bool, default: False
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, q=0.5, rel=False, axis=None, **kwargs):
        super(IndexMassQuantile, self).__init__(*parents, **kwargs)
        self.q = q
        self.rel = rel
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def graph(self, x):
        output = ArgMax(
            (CumSum(Abs(x), axis=self.axis) / Sum(Abs(x), axis=self.axis))
            >= Constant(self.q),
            axis=self.axis,
        ) + Constant(1)
        if self.rel:
            return Graph(output / Length(x, axis=self.axis))
        else:
            return Graph(output)


class Energy(Transformer):
    """
    Sum of the squared values

    Parameters
    ----------
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, axis=None, **kwargs):
        super(Energy, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def graph(self, x):
        return Graph(Sum(Square(x), axis=self.axis))


class EnergyRatio(Transformer):
    """
    Energy ratio

    Divides the values into bins and computes the sum of squares of each bin
    divided by the total sum of squares of all bins.

    Parameters
    ----------
    chunks : int
        Number of bins.
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, chunks=10, axis=None, **kwargs):
        super(EnergyRatio, self).__init__(*parents, **kwargs)
        self.chunks = chunks
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def apply(self, x):
        return apply_to_axis(
            energy_ratio, x, chunks=self.chunks, axis=self.axis
        )


class Entropy(Transformer):
    """
    Entropy

    Parameters
    ----------
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, axis=None, **kwargs):
        super(Entropy, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def apply(self, x):
        return apply_to_axis(entropy, x, axis=self.axis)


class SampleEntropy(Transformer):
    """
    Sample entropy

    Parameters
    ----------
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, axis=None, **kwargs):
        super(SampleEntropy, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def apply(self, x):
        return apply_to_axis(sample_entropy, x, axis=self.axis)


class BinnedDistribution(Transformer):
    """
    Binned distribution

    Divides the values into bins
    and computes the fraction of values in each bin.

    Parameters
    ----------
    bins : int, default: 10
        Number of bins.
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, bins=10, axis=None, **kwargs):
        super(BinnedDistribution, self).__init__(*parents, **kwargs)
        self.bins = bins
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def apply(self, x):
        return apply_to_axis(
            binned_distribution, x, bins=self.bins, axis=self.axis
        )


class BinnedEntropy(Transformer):
    """
    Binned entropy

    Divides the values into bins
    and computes the entropy of the values in each bin.

    Parameters
    ----------
    bins : int, default: 10
         Number of bins.
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, bins=10, axis=None, **kwargs):
        super(BinnedEntropy, self).__init__(*parents, **kwargs)
        self.bins = bins
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def graph(self, x):
        return Graph(
            Entropy(
                BinnedDistribution(x, axis=self.axis, bins=self.bins),
                axis=self.axis,
            )
        )


class C3(Transformer):
    """
    Non-linearity

    Parameters
    ----------
    lag : int, default: 1
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, lag=1, axis=None, **kwargs):
        super(C3, self).__init__(*parents, **kwargs)
        self.lag = lag
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def graph(self, x):
        return Graph(
            Mean(
                Square(Slice(x, i=(2 * self.lag, None), axis=self.axis))
                * Slice(x, i=(self.lag, -self.lag), axis=self.axis)
                * Slice(x, i=(0, -2 * self.lag), axis=self.axis),
                axis=self.axis,
            )
        )


class CID(Transformer):
    """
    Complexity estimate

    Parameters
    ----------
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, axis=None, **kwargs):
        super(CID, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def graph(self, x):
        return Graph(Sqrt(Sum(Square(Diff(x, axis=self.axis)), axis=self.axis)))


class CountAboveMean(Transformer):
    """
    Number of values greater than the mean

    Parameters
    ----------
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, axis=None, **kwargs):
        super(CountAboveMean, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def graph(self, x):
        return Graph(Count(x > Mean(x, axis=self.axis), axis=self.axis))


class CountBelowMean(Transformer):
    """
    Number of values smaller than the mean

    Parameters
    ----------
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, axis=None, **kwargs):
        super(CountBelowMean, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def graph(self, x):
        return Graph(Count(x < Mean(x, axis=self.axis), axis=self.axis))


class RangeCount(Transformer):
    """
    Number of values in range

    Parameters
    ----------
    min : int, default: -1
        Lower bound of the range.
    max : int, default: 1
        Upper bound of the range.
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, min=-1, max=1, axis=None, **kwargs):
        super(RangeCount, self).__init__(*parents, **kwargs)
        self.min = min
        self.max = max
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def graph(self, x):
        return Graph(
            Count(
                (x >= Constant(self.min)) & (x < Constant(self.max)),
                axis=self.axis,
            )
        )


class ValueCount(Transformer):
    """
    Number of occurrences of a value

    Parameters
    ----------
    value : int, default: 0
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, value=0, axis=None, **kwargs):
        super(ValueCount, self).__init__(*parents, **kwargs)
        self.value = value
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def graph(self, x):
        return Graph(Count(Equal(x, Constant(self.value)), axis=self.axis))


class Outliers(Transformer):
    """
    Number of outliers, based on number of unit standard deviations from mean

    Parameters
    ----------
    r : int, default: 3
        Number of unit standard deviations from mean.
    rel : bool, default: False
        Return the number of outliers relative to the total number of values.
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, r=3, rel=False, axis=None, **kwargs):
        super(Outliers, self).__init__(*parents, **kwargs)
        self.r = r
        self.rel = rel
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def graph(self, x):
        n = Count(
            Abs(x - Mean(x, axis=self.axis))
            > StandardDeviation(x, axis=self.axis) * Constant(self.r),
            axis=self.axis,
        )
        if self.rel:
            return Graph(n / Length(x, axis=self.axis))
        else:
            return Graph(n)


class AutoCorrelation(Transformer):
    """
    Auto-correlation coefficients
    
    Returns the auto-correlation coefficients for lags ``[1, ..., t]``
    where ``t`` is the total number of values.

    Parameters
    ----------
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, axis=None, **kwargs):
        super(AutoCorrelation, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def apply(self, x):
        return apply_to_axis(auto_correlation, x, axis=self.axis)


class CrossCorrelation(Transformer):
    """
    Cross-correlation coefficients of two series

    Returns the cross-correlation coefficients for lags ``[1, ..., t]``
    where ``t`` is the total number of values in each series.

    Parameters
    ----------
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, axis=None, **kwargs):
        super(CrossCorrelation, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 2,
            lambda x, y: np.issubdtype(x.dtype, np.float64)
            and np.issubdtype(y.dtype, np.float64),
        ]

    def apply(self, x, y):
        return apply_to_axis(cross_correlation, x, y, axis=self.axis)


class AutoRegressiveCoefficients(Transformer):
    """
    Auto regressive coefficients

    Fits an autoregressive (AR) model and returns the coefficients of the model.

    Parameters
    ----------
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, axis=None, **kwargs):
        super(AutoRegressiveCoefficients, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def apply(self, x):
        def calculator1d(a):
            nnan = ~np.isnan(a)
            a = a[nnan]
            return statsmodels.tsa.ar_model.AutoReg(a, 1).fit().params

        def calculator(a):
            return np.apply_along_axis(calculator1d, -1, a)

        try:
            return apply_to_axis(calculator, x, axis=self.axis)
        except MissingDataError:
            return None


class HighVariance(Transformer):
    """
    Check if the series has a high variance

    Returns true if the variance is larger than a given threshold.

    Parameters
    ----------
    threshold: float, default: 1
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, threshold=1, axis=None, **kwargs):
        super(HighVariance, self).__init__(*parents, **kwargs)
        self.threshold = threshold
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def graph(self, x):
        return Graph(Variance(x, axis=self.axis) > Constant(self.threshold))


class HighStandardDeviation(Transformer):
    """
    Check if the series has a high standard deviation

    Returns true if
    
    .. math::
       StandardDeviation(x) > r * (Max(x) - Min(x))

    Parameters
    ----------
    r : float, default: 1
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, r=1, axis=None, **kwargs):
        super(HighStandardDeviation, self).__init__(*parents, **kwargs)
        self.r = r
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def graph(self, x):
        return Graph(
            StandardDeviation(x, axis=self.axis)
            > (
                (Max(x, axis=self.axis) - Min(x, axis=self.axis))
                * Constant(self.r)
            )
        )


class SymmetryLooking(Transformer):
    """
    Check whether the series looks symmetrical

    Returns true if

    .. math::
       |Mean(x) - Median(x)| < r * (Max(x) - Min(x))

    Parameters
    ----------
    r : float, default: 1
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, r=1, axis=None, **kwargs):
        super(SymmetryLooking, self).__init__(*parents, **kwargs)
        self.r = r
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def graph(self, x):
        g = Graph(
            Abs(Mean(x, axis=self.axis) - Median(x, axis=self.axis))
            < (Max(x, axis=self.axis) - Min(x, axis=self.axis))
            * Constant(self.r)
        )
        return g


class NumberCrossings(Transformer):
    """
    Number of times that the series crosses a given threshold

    Parameters
    ----------
    threshold : float, default: 0
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, threshold=0, axis=None, **kwargs):
        super(NumberCrossings, self).__init__(*parents, **kwargs)
        self.threshold = threshold
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def apply(self, x):
        def calculator(a):
            greater = a > self.threshold
            crossings = np.bitwise_xor(greater[:, :, 1:], greater[:, :, :-1])
            return np.sum(crossings, axis=-1, keepdims=True)

        return apply_to_axis(calculator, x, axis=self.axis)


class LinearTrend(Transformer):
    """
    Linear trend statistics

    Using ``scipy.stats.linregress`` this transformer computes the following 
    linear trend statistics:

    - Slope
    - Intercept
    - Pearson correlation coefficient
    - p-value of the hypothesis test
      where the null hypothesis is that the slope is zero
    - Standard error of the slope
    - Standard error of the intercept

    Parameters
    ----------
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, axis=None, **kwargs):
        super(LinearTrend, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def apply(self, x):
        def calculator1d(a):
            nnan = ~np.isnan(a)
            a = a[nnan]
            return np.array(stats.linregress(np.arange(len(a)), a))

        def calculator(a):
            return np.apply_along_axis(calculator1d, -1, a)

        return apply_to_axis(calculator, x, axis=self.axis)


class LongestStrike(Transformer):
    """
    Largest number of consecutive values not equal to zero

    Parameters
    ----------
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, axis=None, **kwargs):
        super(LongestStrike, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64)
            or np.issubdtype(x.dtype, np.bool_),
        ]

    def apply(self, x):
        return apply_to_axis(longest_non_zero_strike, x, axis=self.axis)


class LongestStrikeAboveMean(Transformer):
    """
    Largest number of consecutive values greather than the mean

    Parameters
    ----------
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, axis=None, **kwargs):
        super(LongestStrikeAboveMean, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def graph(self, x):
        return Graph(LongestStrike(x > Mean(x, axis=self.axis), axis=self.axis))


class LongestStrikeBelowMean(Transformer):
    """
    Largest number of consecutive values smaller than the mean

    Parameters
    ----------
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, axis=None, **kwargs):
        super(LongestStrikeBelowMean, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def graph(self, x):
        return Graph(LongestStrike(x < Mean(x, axis=self.axis), axis=self.axis))


class SumChange(Transformer):
    """
    Sum of differences between consecutive timestamps

    Parameters
    ----------
    abs : bool, default: False
        Take the absolute value of the differences
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, abs=False, axis=None, **kwargs):
        super(SumChange, self).__init__(*parents, **kwargs)
        self.abs = abs
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def graph(self, x):
        if self.abs:
            return Graph(Sum(Abs(Diff(x, axis=self.axis)), axis=self.axis))
        else:
            return Graph(Sum(Diff(x, axis=self.axis), axis=self.axis))


class MeanChange(Transformer):
    """
    Average of differences between consecutive timestamps

    Parameters
    ----------
    abs : bool, default: False
        Take the absolute value of the differences
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, abs=False, axis=None, **kwargs):
        super(MeanChange, self).__init__(*parents, **kwargs)
        self.abs = abs
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def graph(self, x):
        if self.abs:
            return Graph(Mean(Abs(Diff(x, axis=self.axis)), axis=self.axis))
        else:
            return Graph(Mean(Diff(x, axis=self.axis), axis=self.axis))


class MeanSecondDerivativeCentral(Transformer):
    """
    Mean of the central approximation of the second derivative

    Parameters
    ----------
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, axis=None, **kwargs):
        super(MeanSecondDerivativeCentral, self).__init__(*parents, **kwargs)
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def graph(self, x):
        return Graph(
            Mean(
                Slice(x, i=(2, None), axis=self.axis)
                - Slice(x, i=(1, -1), axis=self.axis) * Constant(2)
                + Slice(x, i=(0, -2), axis=self.axis),
                axis=self.axis,
            )
        )


class TimeReversalAsymmetryStatistic(Transformer):
    """
    Time reversal asymmetry statistic

    Parameters
    ----------
    lag : int, default: 1
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, lag=1, axis=None, **kwargs):
        super(TimeReversalAsymmetryStatistic, self).__init__(*parents, **kwargs)
        self.lag = lag
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
            lambda x: length(x, self.axis) > self.lag,
        ]

    def graph(self, x):
        return Graph(
            Mean(
                Square(Slice(x, i=(2 * self.lag, None), axis=self.axis))
                * Slice(x, i=(self.lag, -self.lag), axis=self.axis)
                - Slice(x, i=(self.lag, -self.lag), axis=self.axis)
                * Square(Slice(x, i=(0, -2 * self.lag), axis=self.axis)),
                axis=self.axis,
            )
        )


class FriedrichCoefficients(Transformer):
    """
    Coefficients of a polynomial fitted to the deterministic dynamics of 
    a Langevin model.

    Parameters
    ----------
    m : int, default: 1
    r : int, default: 10
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, m=1, r=10, axis=None, **kwargs):
        super(FriedrichCoefficients, self).__init__(*parents, **kwargs)
        self.m = m
        self.r = r
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def apply(self, x):
        def calculator1d(a):
            nnan = ~np.isnan(a)
            a = a[nnan]
            signal = a[:-1]
            delta = np.diff(a)
            xy = np.empty((self.r, 2))
            quantiles = np.linspace(0, 1, self.r + 1)
            for i in range(self.r):
                q_min = np.quantile(signal, quantiles[i])
                q_max = np.quantile(signal, quantiles[i + 1])
                if i == 0:
                    q = signal <= q_max
                elif i == self.r - 1:
                    q = signal > q_min
                else:
                    q = (signal > q_min) & (signal <= q_max)
                xy[i, 0] = np.mean(signal[q])
                xy[i, 1] = np.mean(delta[q])
            xy = xy[~np.isnan(xy).any(axis=1)]
            return np.polyfit(xy[:, 0], xy[:, 1], deg=self.m)

        def calculator(a):
            return np.apply_along_axis(calculator1d, -1, a)

        return apply_to_axis(calculator, x, axis=self.axis)


class MaxLangevinFixedPoint(Transformer):
    """
    Largest fixed point of a polynomial fitted to the deterministic dynamics of
    a Langevin model.

    Parameters
    ----------
    m : int, default: 1
    r : int, default: 10
    axis : {'time', 'dims'}, optional
        Direction of time: timestamps ('time') or dimensions ('dims').
        Default: first axis with more than one value.
    """

    def __init__(self, *parents, m=1, r=10, axis=None, **kwargs):
        super(MaxLangevinFixedPoint, self).__init__(*parents, **kwargs)
        self.m = m
        self.r = r
        self.axis = axis
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
        ]

    def graph(self, x):
        return Graph(
            Max(
                Roots(
                    FriedrichCoefficients(
                        x, m=self.m, r=self.r, axis=self.axis
                    ),
                    axis=self.axis,
                ),
                axis=self.axis,
            )
        )
