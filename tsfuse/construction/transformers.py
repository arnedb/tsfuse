import numpy as np

from ..transformers import *

__all__ = [
    'minimal',
    'fast',
    'full',
]

minimal = {
    'series-to-series': [
        Resultant(),
        Ratio(),
        Difference(rel=True),
        Difference(rel=False),
    ],
    'series-to-attribute': [
        Length(),
        Sum(),
        Min(),
        Max(),
        Mean(),
        Median(),
        StandardDeviation(),
        Variance(),
        Skewness(),
        Kurtosis(),
    ]
}

fast = {
    'series-to-series':
        minimal['series-to-series'],
    'series-to-attribute':
        minimal['series-to-attribute'] + [
            # BinnedDistribution(bins=10),
            BinnedEntropy(bins=10),
        ] + [
            C3(lag=lag) for lag in (1, 2, 3)
        ] + [
            CID(),
        ] + [
            CountAboveMean(),
            CountBelowMean(),
        ] + [
            Energy(),
            EnergyRatio(chunks=10),
        ] + [
            SumChange(abs=True),
        ] + [
            MeanChange(abs=abs)
            for abs in (True, False)
        ] + [
            MeanSecondDerivativeCentral(),
        ] + [
            IndexMassQuantile(q=round(q, 1), rel=True)
            for q in (.1, .2, .3, .4, .6, .7, .8, .9)
        ] + [
            ArgMin(first=first, rel=True)
            for first in (True, False)
        ] + [
            ArgMax(first=first, rel=True)
            for first in (True, False)
        ] + [
        #     NumberPeaks(support=support)
        #     for support in (1, 3, 5, 10, 50)
        # ] + [
            NumberCrossings(threshold=-1),
            NumberCrossings(threshold=0),
            NumberCrossings(threshold=1),
        ] + [
            LongestStrikeAboveMean(),
            LongestStrikeBelowMean(),
        ] + [
            TimeReversalAsymmetryStatistic(lag=lag)
            for lag in (1, 2, 3)
        ] + [
            HighStandardDeviation(r=round(r, 1))
            for r in (.1, .2, .3, .4, .6, .7, .8, .9)
        ] + [
            HighVariance(),
        ] + [
            SymmetryLooking(r=round(r, 1))
            for r in (.1, .2, .3, .4, .6, .7, .8, .9)
        ] + [
            RangeCount(min=-1, max=1),
            RangeCount(min=-np.inf, max=0),
            RangeCount(min=0, max=np.inf),
        ] + [
            ValueCount(value=-1),
            ValueCount(value=0),
            ValueCount(value=1),
        ] + [
            Outliers(r=r, rel=rel)
            for r in (1, 1.5, 2, 3, 4, 5)
            for rel in (True, False)
        ] + [
            HasDuplicateMin(),
            HasDuplicateMax(),
        ] + [
            Slice(AutoCorrelation(), i=i, axis='timestamps') for i in range(10)
        ]
}

full = {
    'series-to-series':
        fast['series-to-series'],
    'series-to-attribute':
        fast['series-to-attribute'] + [
            Quantile(q=round(q, 1))
            for q in (.1, .2, .3, .4, .6, .7, .8, .9)
        ] + [
            Slice(FFT(), i=i, axis='timestamps') for i in range(100)
        ] + [
            Slice(CWT(), i=i, axis='timestamps') for i in range(10)
        ] + [
            SpectralMean(FFT()),
            SpectralVariance(FFT()),
            SpectralSkewness(FFT()),
            SpectralKurtosis(FFT()),
        ] + [
            PowerSpectralDensity(),
        ] + [
            LinearTrend(Aggregate(size=size, agg=agg))
            for size in (5, 10, 50)
            for agg in ('mean', 'var', 'min', 'max')
        ] + [
            Slice(AutoRegressiveCoefficients(), i=i, axis='timestamps') for i in range(10)
        ] + [
            Slice(FriedrichCoefficients(m=3, r=30), i=i, axis='timestamps') for i in range(4)
        ] + [
            MaxLangevinFixedPoint(m=3, r=30),
        ] + [
            NumberPeaksCWT(),
        ] + [
            LinearTrend(),
        ] + [
            NumberUniqueValues(),
            SumReoccurringValues(),
            SumReoccurringDataPoints(),
            HasDuplicate(),
        ]
}
