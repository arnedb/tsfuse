import itertools
import warnings
import copy
import time
from math import inf

import numpy as np

from ..computation.util import to_dataframe
from ..data import Collection
from ..computation import Input, Graph
from ..transformers import *

__all__ = [
    'construct',
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
            Slice(AutoCorrelation(), i=i, axis='time') for i in range(10)
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
            Slice(FFT(), i=i, axis='time') for i in range(100)
        ] + [
            Slice(CWT(), i=i, axis='time') for i in range(10)
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
            Slice(AutoRegressiveCoefficients(), i=i, axis='time') for i in range(10)
        ] + [
            Slice(FriedrichCoefficients(m=3, r=30), i=i, axis='time') for i in range(4)
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


def construct(X, transformers='full', max_depth=1, corr=0.99,
              return_data=False, return_log=False):
    # Create a log during the construction process
    log = {time.time(): {'event': 'started'}} if return_log else None

    # Format arguments
    X, transformers, max_depth = format_args(X, transformers, max_depth)
    N = len(X[list(X)[0]])

    # Step 1: series-to-series transformations
    if log: log[time.time()] = {'event': 'series-to-series-started'}
    series = build_series(X, N, transformers['series-to-series'], max_depth, corr, log=log)
    if log: log[time.time()] = {'event': 'series-to-series-finished'}

    # Step 2: series-to-attribute transformations
    if log: log[time.time()] = {'event': 'series-to-attribute-started'}
    s2a = transformers['series-to-attribute']
    attributes, data = build_attributes(X, series, s2a, log=log)
    if log: log[time.time()] = {'event': 'series-to-attribute-finished'}

    # Create result
    if log: log[time.time()] = {'event': 'finished'}
    return create_result(attributes, data, log, return_data, return_log)


def build_series(X, N, transformers, max_depth, corr, log=None):
    series_depth_0_to_d = []
    series_depth_d = []
    views = list(X)
    data = {Input(view).trace: X[view] for view in X}
    stats = []

    depth = 0
    while True:
        new_series = []
        for transformer in generate_series_to_series_transformers(
                views, transformers, data, series_depth_0_to_d, series_depth_d, depth
        ):
            if log: log[time.time()] = {
                'event': 'series-to-series-try',
                'transformer': str(transformer),
            }

            # Compute output of the transformer
            if isinstance(transformer, Input):
                output = data[transformer.trace]
            else:
                # output = transformer.transform(*[data[p.trace] for p in transformer.parents])
                try:
                    result = Graph(transformer).transform(X, return_dataframe=False)
                    output = result[list(result)[0]]
                except:
                    output = None
            if output is None:
                continue

            # Compute stats for the transformer
            tstats = to_collection(SinglePassStatistics().transform(output)).values  # .reshape((N, -1))

            # Check redundancy
            non_redundant = check_non_redundant(tstats, stats, corr)

            # Add if non redundant
            if non_redundant:
                # data[transformer.trace] = output
                new_series.append(transformer)
                stats.append(tstats)
                if log: log[time.time()] = {
                    'event': 'series-to-series-add',
                    'transformer': str(transformer),
                }

            # Skip otherwise
            elif log:
                log[time.time()] = {
                    'event': 'series-to-series-redundant',
                    'transformer': str(transformer),
                }

        series_depth_0_to_d = series_depth_0_to_d + series_depth_d
        series_depth_d = new_series

        # Stop early when no new series could be added
        if len(series_depth_d) == 0:
            break

        depth = depth + 1
        if depth > max_depth:
            break

    # Return series (constructed transformers) and their corresponding data (computed values)
    return series_depth_0_to_d + series_depth_d


def build_attributes(X, series, transformers, log=None):
    attributes = []
    attributes_data = dict()

    for s in series:
        if isinstance(s, Input):
            x = X[s.input_id]
        else:
            result = Graph(s).transform(X, return_dataframe=False)
            x = result[list(result)[0]]
        for t in transformers:
            transformer = copy.deepcopy(t)
            transformer._parents = [s]

            if log: log[time.time()] = {
                'event': 'series-to-attribute-try',
                'transformer': str(transformer),
            }

            # xs, ys = subsample(x, y, ratio=0.1)

            # Compute output of the transformer
            try:
                output = to_collection(transformer.transform(x))
            except:
                output = None
            if output is None:
                continue

            # Relevant
            if log: log[time.time()] = {
                'event': 'series-to-attribute-add',
                'transformer': transformer,
            }

            attributes.append(transformer)
            attributes_data[transformer.trace] = output
            # attributes_data[transformer.trace] = None

    # Return attributes (constructed transformers) and the data
    return attributes, attributes_data


def subsample(x, y, ratio=0.1):
    i = np.random.choice(x.shape[0], size=int(ratio * x.shape[0]))
    xs = Collection.from_array(x.values[i], dims=x.dims, time=x.time[i])
    ys = y[i]
    return xs, ys


def generate_series_to_series_transformers(views, transformers, data, series_depth_0_to_d,
                                           series_depth_d, depth):
    if depth < 1:
        for view in views:
            yield Input(view)
    else:
        series = series_depth_0_to_d + series_depth_d
        for t in transformers:
            for parents in itertools.combinations(series, t.n_inputs):
                if not any(p in series_depth_d for p in parents):
                    continue
                if not constraints_satisfied(parents, data):
                    continue
                for transformer in set_parents(t, parents, data):
                    yield transformer


def constraints_satisfied(parents, data):
    if len(parents) == 2:
        if isinstance(parents[0], Input) and isinstance(parents[1], Input):
            data1 = data[parents[0].trace]
            data2 = data[parents[1].trace]
            if hasattr(data1, 'tags') and hasattr(data2, 'tags'):
                if ('location' in data1.tags) and ('location' in data2.tags) \
                        and ('type' in data1.tags) and ('type' in data2.tags):
                    return ((data1.tags['location'] == data2.tags['location']) or
                            (data1.tags['type'] == data2.tags['type']))
    return True


def check_non_redundant(tstats, stats, corr):
    if len(stats) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # r = np.corrcoef(np.concatenate([stats, tstats.T], axis=0))
            # r = np.abs(r[:stats.shape[0], stats.shape[0]:])
            # return not np.any(r > corr)
            for other in stats:
                for j in range(tstats.shape[2]):
                    for k in range(other.shape[2]):
                        for i in range(2, 8):
                            if np.corrcoef(tstats[:, i, j], other[:, i, k])[0][1] > corr:
                                return False
            return True
    else:
        return True


def set_parents(node, parents, data):
    """
    Returns
    -------
    list(Transformer)
    """
    c = copy.deepcopy(node)
    c._parents = parents
    return [c]
    # n_timestamps = [data[p.trace].shape[1] for p in parents]
    # # Resample parents if number timestamps is not unique
    # # and return a transformer for each sample rate
    # if len(set(n_timestamps)) > 1:
    #     transformers = []
    #     for num in set(n_timestamps):
    #         resampled = []
    #         for i, parent in enumerate(parents):
    #             if n_timestamps[i] != num:
    #                 resampled.append(Resample(parent, num=num, axis='time'))
    #             else:
    #                 resampled.append(parent)
    #         c = copy.deepcopy(node)
    #         c._parents = resampled
    #         transformers.append(c)
    #     return transformers
    # # Otherwise, use the original parents
    # else:
    #     c = copy.deepcopy(node)
    #     c._parents = parents
    #     return [c]


# def create_graph(attributes):
#     graph = Graph()
#     nodes = []
#     for attr in attributes:
#         nodes.append(graph.add_node(attr))
#     for node in graph.nodes:
#         node._is_output = False
#     for node in nodes:
#         node._is_output = True
#     return graph


def create_result(attributes, data, log, return_data, return_log):
    result = list()
    graph = Graph()
    result.append(graph)
    nodes = []
    if return_data:
        outputs = dict()
        # result.append({node: data[attributes[i].trace] for i, node in enumerate(graph.outputs)})
        for attribute in attributes:
            node = graph.add_node(attribute)
            nodes.append(node)
            outputs[node] = data[attribute.trace]
        result.append(to_dataframe(outputs))
    else:
        for attr in attributes:
            nodes.append(graph.add_node(attr))
    for node in graph.nodes:
        node._is_output = False
    for node in nodes:
        node._is_output = True
    if return_log:
        result.append(log)
    if len(result) == 1:
        result = result[0]
    else:
        result = tuple(result)
    return result


def format_args(X, transformers, max_depth):
    # X should be a dict(int or str: Collection)
    if isinstance(X, list):
        X = {i: X[i] for i in range(len(X))}
    elif isinstance(X, Collection):
        X = {'X': X}
    elif not isinstance(X, dict):
        X = {'X': Collection.from_array(X)}
    # Transformers should be {'series-to-series': list, 'series-to-attributes': list}
    if transformers == 'minimal':
        t = minimal
    elif transformers == 'fast':
        t = fast
    elif transformers == 'full':
        t = full
    else:
        t = copy.deepcopy(transformers)
    # Max. depth should be a number or inf
    if isinstance(max_depth, int) or isinstance(max_depth, float):
        if max_depth < 0:
            max_depth = inf
        else:
            max_depth = int(max_depth)
    else:
        max_depth = inf
    return X, t, max_depth


def flatten(values):
    if values.ndim == 2:
        return values[:, 0]
    elif values.ndim == 3:
        return values[:, 0, 0]
    else:
        return values.flatten()


def to_collection(x):
    if x is None:
        return None
    elif isinstance(x, Collection):
        return x
    else:
        if any(c is None for c in x):
            return None
        elif len(set([c.shape[1] for c in x])) > 1:
            return None
        else:
            return Collection.from_array(np.concatenate([c.values for c in x]))
