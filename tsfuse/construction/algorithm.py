import itertools
import warnings
import random

from functools import partial

import numpy as np
import pandas as pd

from tsfuse.errors import InvalidPreconditionError
from tsfuse.computation import Graph, Input
from tsfuse.computation.util import to_dataframe
from tsfuse.transformers import *

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn_gbmi import *


class TSFuseExtractor(object):
    def __init__(
        self,
        transformers="full",
        max_depth=3,
        series_fusion=True,
        attribute_fusion=True,
        max_series_permutations=10,
        max_attribute_permutations=1000,
        max_series_correlation=0.9,
        coefficients=0.1,
        interaction=0.1,
        compatible=lambda x, y: True,
        task="auto",
        verbose=False,
        random_state=None,
    ):
        self.transformers = create(transformers)
        self.max_depth = max_depth
        self.series_fusion = series_fusion
        self.attribute_fusion = attribute_fusion
        self.max_series_permutations = max_series_permutations
        self.max_attribute_permutations = max_attribute_permutations
        self.max_series_correlation = max_series_correlation
        self.coefficients = coefficients
        self.interaction = interaction
        self.compatible = compatible
        self.task = task
        self.verbose = verbose
        self.random_state = random_state

    def transform(self, X):
        return self.graph_.transform(X)

    def fit(self, X, y):
        assert y.isna().sum() == 0, "y must not contain any NaN values"
        self.task_ = self.detect_task(y) if self.task == "auto" else self.task
        data = self.initialize(X, y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.series_to_series(data)
            self.series_to_attributes(data)
            self.attributes_to_attributes(data, y)
            self.select_attributes(data, y)
        for n in self.selected_attributes_:
            n = self.graph_.add_node(n)
            n._is_output = True
        return self

    def fit_transform(self, X, y):
        assert y.isna().sum() == 0, "y must not contain any NaN values"
        self.task_ = self.detect_task(y) if self.task == "auto" else self.task
        data = self.initialize(X, y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.series_to_series(data)
            self.series_to_attributes(data)
            self.attributes_to_attributes(data, y)
            self.select_attributes(data, y)
        output = dict()
        for n in self.selected_attributes_:
            n = self.graph_.add_node(n)
            n._is_output = True
            output[n] = data[n.trace]
        return to_dataframe(output)

    def initialize(self, X, y):
        self.graph_ = Graph()
        self.depth_ = dict()
        data = dict()
        for v in X:
            # Create an input node
            i = Input(v)
            # Add the node to the graph
            self.graph_.add_node(i)
            # Keep track of depths of nodes
            self.depth_[i.trace] = 0
            # Initialize input data
            data[i.trace] = X[v]
        # Done
        return data

    def series_to_series(self, data):
        # Start from input series
        series = [self.graph_.inputs[i] for i in self.graph_.inputs]
        # Generate series for increasing depths
        if self.series_fusion:
            transformers = self.transformers["series-to-series-nofusion"]
            transformers += self.transformers["series-to-series-fusion"]
            for depth in range(self.max_depth - 1):
                generated = []
                nodes_c = [n for n in series if self.depth_[n.trace] < depth]
                nodes_d = [n for n in series if self.depth_[n.trace] == depth]
                # Generate nodes for all transformers
                p_prev = 0
                for t in sorted(transformers, key=self.num_parents):
                    p = self.num_parents(t)
                    if p != p_prev:
                        # Generate compatible permutations
                        permutations = []
                        for permutation in self.generate_permutations(
                            nodes_c, nodes_d, p
                        ):
                            if (p > 1) and (
                                len(permutations)
                                >= self.max_series_permutations
                            ):
                                break
                            if self.check_compatible(data, permutation):
                                permutations.append(permutation)
                        # Create transformer node for each permutation
                        for permutation in permutations:
                            transformer = t(*permutation)
                            # Check preconditions
                            try:
                                transformer.check_preconditions(
                                    *[data[n.trace] for n in permutation]
                                )
                            except InvalidPreconditionError:
                                continue
                            # Compute output
                            output = transformer.transform(
                                *[data[n.trace] for n in permutation]
                            )
                            # Checked, now add the transformer to the generated nodes
                            if output is not None:
                                generated.append(transformer)
                                data[transformer.trace] = output
                                self.depth_[transformer.trace] = depth + 1
                # Done for depth
                series += generated
        # Transform to one-dimensional individual series
        series = self.to1d(data, series)
        # Select non-redundant series
        series = self.select_non_redundant_series(
            data, series, corr=self.max_series_correlation
        )
        # Done
        self.series_ = series

    def series_to_attributes(self, data):
        extracted = []
        series = []
        for s in self.series_:
            x = data[s.trace]
            if x.shape[2] == 1:
                series.append(s)
        # Series-to-attribute transformers
        transformers = self.transformers["series-to-attribute-nofusion"]
        if self.series_fusion:
            transformers += self.transformers["series-to-attribute-fusion"]
        p_prev = 0
        for t in sorted(transformers, key=self.num_parents):
            p = self.num_parents(t)
            if p != p_prev:
                # Generate compatible permutations
                permutations = []
                for permutation in self.generate_permutations([], series, p):
                    if (p > 1) and (
                        len(permutations) >= self.max_series_permutations
                    ):
                        break
                    if self.check_compatible(data, permutation):
                        permutations.append(permutation)
                # Create transformer node for each permutation
                for permutation in permutations:
                    transformer = t(*permutation)
                    # Check preconditions
                    try:
                        transformer.check_preconditions(
                            *[data[n.trace] for n in permutation]
                        )
                    except InvalidPreconditionError:
                        continue
                    # Compute output
                    output = transformer.transform(
                        *[data[n.trace] for n in permutation]
                    )
                    if output is not None:
                        self.depth_[transformer.trace] = (
                            max([self.depth_[s.trace] for s in permutation]) + 1
                        )
                        extracted.append(transformer)
                        data[transformer.trace] = output
        # Series-to-coefficients transformers
        transformers = self.transformers["series-to-coefficients-nofusion"]
        if self.series_fusion:
            transformers += self.transformers["series-to-coefficients-fusion"]
        p_prev = 0
        for t in sorted(transformers, key=self.num_parents):
            p = self.num_parents(t)
            if p != p_prev:
                # Generate compatible permutations
                permutations = []
                for permutation in self.generate_permutations([], series, p):
                    if (p > 1) and (
                        len(permutations) >= self.max_series_permutations
                    ):
                        break
                    if self.check_compatible(data, permutation):
                        permutations.append(permutation)
                # Create transformer node for each permutation
                for permutation in permutations:
                    transformer = t(*permutation)
                    # Check preconditions
                    try:
                        transformer.check_preconditions(
                            *[data[n.trace] for n in permutation]
                        )
                    except InvalidPreconditionError:
                        continue
                    # Compute output
                    output = transformer.transform(
                        *[data[n.trace] for n in permutation]
                    )
                    if output is not None:
                        # Select largest coefficients
                        for c, o in self.largest_coefficients(
                            transformer, output, self.coefficients
                        ):
                            self.depth_[c.trace] = (
                                max([self.depth_[s.trace] for s in permutation])
                                + 1
                            )
                            extracted.append(c)
                            data[c.trace] = o
        # Done
        self.extracted_attributes_ = extracted

    def attributes_to_attributes(self, data, y):
        # Start from extracted attributes
        attributes = self.extracted_attributes_[:]
        self.generated_attributes_ = []
        # Generate attributes for increasing depths
        if self.attribute_fusion:
            transformers = self.transformers["attribute-to-attribute-nofusion"]
            transformers += self.transformers["attribute-to-attribute-fusion"]
            for depth in range(1, self.max_depth):
                # Init
                generated = []
                nodes_c = [
                    n for n in attributes if self.depth_[n.trace] < depth
                ]
                nodes_d = [
                    n for n in attributes if self.depth_[n.trace] == depth
                ]
                # Generate nodes for all transformers
                p_prev = 0
                for t in sorted(transformers, key=self.num_parents):
                    p = self.num_parents(t)
                    if p != p_prev:
                        # Generate compatible permutations
                        permutations = []
                        for permutation in self.generate_permutations(
                            nodes_c, nodes_d, p
                        ):
                            if (p > 1) and (
                                len(permutations)
                                >= self.max_attribute_permutations
                            ):
                                break
                            if self.check_compatible(data, permutation):
                                permutations.append(permutation)
                        # Select interacting permutations
                        permutations = self.select_interactions(
                            data, y, permutations
                        )
                    # Create transformer node for each permutation
                    for permutation in permutations:
                        transformer = t(*permutation)
                        # Check preconditions
                        try:
                            transformer.check_preconditions(
                                *[data[n.trace] for n in permutation]
                            )
                        except InvalidPreconditionError:
                            continue
                        # Compute output
                        output = transformer.transform(
                            *[data[n.trace] for n in permutation]
                        )
                        # Checked, now add the transformer to the generated nodes
                        if output is not None:
                            generated.append(transformer)
                            data[transformer.trace] = output
                            self.depth_[transformer.trace] = depth + 1
                # Done for depth
                attributes += generated
                self.generated_attributes_ += generated

    def select_attributes(self, data, y):
        # Collect all extracted and generated attributes
        attributes = self.extracted_attributes_ + self.generated_attributes_
        # Group similar features

        def p(node):
            parents = ",".join([p(parent) for parent in node.parents])
            return "{}({})".format(node.__class__.__name__, parents)

        patterns = dict()
        for transformer in attributes:
            pattern = p(transformer)
            if pattern not in patterns:
                patterns[pattern] = [transformer]
            else:
                patterns[pattern].append(transformer)
        # Select per group
        selected = []
        for p in patterns:
            # Create DataFrame with feature values
            X = pd.DataFrame(index=np.arange(len(y)))
            for transformer in patterns[p]:
                output = data[transformer.trace]
                X[transformer] = output.values[:, 0, 0]
            X = (
                X.astype("float32")
                .replace([np.inf, -np.inf], np.nan)
                .dropna(axis=1)
            )
            if X.shape[1] == 0:
                continue
            # Train decision tree model
            if self.task_ == "classification":
                model = DecisionTreeClassifier(random_state=self.random_state)
            else:
                model = DecisionTreeRegressor(random_state=self.random_state)
            model.fit(X, y)
            # Select features used in the tree
            selected += list(X.columns[model.feature_importances_ > 0])
        # Done
        self.selected_attributes_ = selected

    def generate_permutations(self, nodes_c, nodes_d, p):
        nodes_c = nodes_c[:]
        nodes_d = nodes_d[:]
        if self.random_state is not None:
            random.seed(self.random_state)
        random.shuffle(nodes_d)

        def generate(x, other, n):
            if n > 1:
                for permutation in itertools.permutations(other, n - 1):
                    for i in range(n):
                        yield list(permutation[:i]) + [x] + list(
                            permutation[i:]
                        )
            else:
                yield [x]

        for i, node in enumerate(nodes_d):
            if self.random_state is not None:
                random.seed(self.random_state)
            random.shuffle(nodes_c)
            for permutation in generate(node, nodes_c + nodes_d[:i], p):
                yield permutation

    def generate_permutations_attr(self, nodes, p):
        nodes = nodes[:]
        if self.random_state is not None:
            random.seed(self.random_state)
        random.shuffle(nodes)

        if p > 1:
            for permutation in itertools.permutations(nodes, p):
                yield permutation

    def check_compatible(self, data, permutation):
        if len(permutation) == 1:
            return True
        for i in range(len(permutation)):
            x = data[permutation[i].trace]
            c = False
            for j in range(len(permutation)):
                if i == j:
                    continue
                y = data[permutation[j].trace]
                if self.compatible(x, y) or self.compatible(y, x):
                    c = True
            if not c:
                return False
        return True

    def select_non_redundant_series(self, data, generated, corr=0.9):
        stats = []
        selected = []

        def check_non_redundant_series(transformer_stats):
            if len(stats) > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for other in stats:
                        for j in range(transformer_stats.shape[2]):
                            for k in range(other.shape[2]):
                                for i in range(
                                    2, 8
                                ):  # don't use length and sum
                                    c = abs(
                                        np.corrcoef(
                                            transformer_stats[:, i, j],
                                            other[:, i, k],
                                        )[0][1]
                                    )
                                    if c > corr:
                                        return False
                    return True
            else:
                return True

        # Select series one by one
        for node in generated:
            output = data[node.trace]
            # Extract simple statistics
            transformer_stats = SinglePassStatistics().transform(output).values
            # Check correlation with other series
            if check_non_redundant_series(transformer_stats):
                selected.append(node)
                stats.append(transformer_stats)
            else:
                del data[node.trace]

        # Done
        return selected

    def select_interactions(self, data, y, permutations):
        # Collect data for the nodes in the permutations only
        permutations_nodes = set()
        for permutation in permutations:
            for node in permutation:
                permutations_nodes.add(node)
        # Convert to DataFrame and drop columns with inf/-inf/nan values
        if len(permutations_nodes) == 0:
            return []
        data_permutations = to_dataframe(
            {n: data[n.trace] for n in permutations_nodes}
        )
        data_permutations = data_permutations.replace(
            [np.inf, -np.inf], np.nan
        ).dropna(axis=1)
        data_permutations = data_permutations.astype("float32")
        data_permutations = data_permutations.loc[
            :, np.isfinite(data_permutations.values).all(axis=0)
        ]
        if data_permutations.shape[1] == 0:
            return []
        # Fit a gradient boosted tree model using sklearn
        if self.task_ == "classification":
            gbr = GradientBoostingClassifier(
                random_state=self.random_state
            ).fit(data_permutations, y)
        else:
            gbr = GradientBoostingRegressor(random_state=self.random_state).fit(
                data_permutations, y
            )
        # Compute H-values
        hvalues = pd.Series(index=[str(p) for p in permutations])
        for permutation in permutations:
            if all(str(n) in data_permutations.columns for n in permutation):
                hvalue = h(
                    gbr, data_permutations.loc[:, [str(n) for n in permutation]]
                )
                hvalues.loc[str(permutation)] = hvalue
        # Select permutations with large enough H-values
        selected = hvalues.loc[hvalues > self.interaction]
        return [p for p in permutations if str(p) in selected.index]

    def detect_task(self, y):
        if np.issubdtype(y, np.float64):
            return "regression"
        else:
            return "classification"

    def num_parents(self, transformer):
        if hasattr(transformer, "__code__"):
            return transformer.__code__.co_argcount
        else:
            return 1

    def to1d(self, data, generated):
        generated1d = []
        for n in generated:
            x = data[n.trace]
            if x.shape[2] == 1:
                generated1d.append(n)
            else:
                for i in range(x.shape[2]):
                    s = Slice(n, i=i, axis="dims")
                    self.depth_[s.trace] = self.depth_[n.trace]
                    data[s.trace] = s.transform(x)
                    generated1d.append(s)
        return generated1d

    def largest_coefficients(self, transformer, output, coeff):
        N = np.min(output.shape[1])
        s = Slice(i=slice(0, N), axis="time").transform(output)
        s = Abs().transform(s)
        s = Sum(axis="windows").transform(s)
        s = s.values[0, :, 0]
        n_coeff = int(len(s) * coeff)
        for i in np.argsort(s)[-n_coeff:][::-1]:
            t = Slice(transformer, i=int(i), axis="time")
            o = t.transform(output)
            if o is None:
                continue
            else:
                yield t, o


def create(setting):

    # For multiple transformers
    single_dimension = lambda *collections: all(
        c.shape[2] == 1 for c in collections
    )

    # For Ratio transformer
    non_zero_second_argument = lambda x, y: np.sum(y.values == 0) == 0

    minimal = {
        "series-to-series-nofusion": [lambda x: Diff(x),],
        "series-to-series-fusion": [
            lambda x: Resultant(x),
            lambda x, y: Average(x, y, with_preconditions=[single_dimension]),
            lambda x, y: Difference(
                x, y, with_preconditions=[single_dimension]
            ),
            lambda x, y: Ratio(
                x,
                y,
                with_preconditions=[single_dimension, non_zero_second_argument],
            ),
            lambda a, b, c: Angle(a, b, c),
        ],
        "attribute-to-attribute-nofusion": [
            lambda x: Reciprocal(x, with_preconditions=[single_dimension]),
        ],
        "attribute-to-attribute-fusion": [
            lambda x, y: Average(x, y, with_preconditions=[single_dimension]),
            lambda x, y: Difference(
                x, y, with_preconditions=[single_dimension]
            ),
            lambda x, y: Ratio(
                x,
                y,
                with_preconditions=[single_dimension, non_zero_second_argument],
            ),
        ],
        "series-to-attribute-nofusion": [
            lambda x: Mean(
                x, with_preconditions=[single_dimension], axis="time"
            ),
            lambda x: Median(
                x, with_preconditions=[single_dimension], axis="time"
            ),
            lambda x: Variance(
                x, with_preconditions=[single_dimension], axis="time"
            ),
            lambda x: StandardDeviation(
                x, with_preconditions=[single_dimension], axis="time"
            ),
            lambda x: Skewness(
                x, with_preconditions=[single_dimension], axis="time"
            ),
            lambda x: Kurtosis(
                x, with_preconditions=[single_dimension], axis="time"
            ),
            lambda x: Min(
                x, with_preconditions=[single_dimension], axis="time"
            ),
            lambda x: Max(
                x, with_preconditions=[single_dimension], axis="time"
            ),
            lambda x: Sum(
                x, with_preconditions=[single_dimension], axis="time"
            ),
            lambda x: Length(
                x, with_preconditions=[single_dimension], axis="time"
            ),
        ],
        "series-to-attribute-fusion": [],
        "series-to-coefficients-nofusion": [],
        "series-to-coefficients-fusion": [],
    }

    full = {
        "series-to-series-nofusion": minimal["series-to-series-nofusion"],
        "series-to-series-fusion": minimal["series-to-series-fusion"],
        "attribute-to-attribute-nofusion": minimal[
            "attribute-to-attribute-nofusion"
        ],
        "attribute-to-attribute-fusion": minimal[
            "attribute-to-attribute-fusion"
        ],
        "series-to-attribute-nofusion": [
            lambda x: Mean(
                x, with_preconditions=[single_dimension], axis="time"
            ),
            lambda x: Median(
                x, with_preconditions=[single_dimension], axis="time"
            ),
            lambda x: Variance(
                x, with_preconditions=[single_dimension], axis="time"
            ),
            lambda x: StandardDeviation(
                x, with_preconditions=[single_dimension], axis="time"
            ),
            lambda x: Skewness(
                x, with_preconditions=[single_dimension], axis="time"
            ),
            lambda x: Kurtosis(
                x, with_preconditions=[single_dimension], axis="time"
            ),
            lambda x: Min(
                x, with_preconditions=[single_dimension], axis="time"
            ),
            lambda x: Max(
                x, with_preconditions=[single_dimension], axis="time"
            ),
            lambda x: Sum(
                x, with_preconditions=[single_dimension], axis="time"
            ),
            lambda x: Length(
                x, with_preconditions=[single_dimension], axis="time"
            ),
            lambda x: Energy(
                x, with_preconditions=[single_dimension], axis="time"
            ),
            lambda x: C3(x, with_preconditions=[single_dimension], axis="time"),
            lambda x: CID(
                x, with_preconditions=[single_dimension], axis="time"
            ),
            lambda x: CountAboveMean(
                x, with_preconditions=[single_dimension], axis="time"
            ),
            lambda x: CountBelowMean(
                x, with_preconditions=[single_dimension], axis="time"
            ),
        ]
        + [
            partial(
                lambda x, q: Quantile(
                    x, q=q, with_preconditions=[single_dimension], axis="time",
                ),
                q=q,
            )
            for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        ]
        + [
            partial(
                lambda x, i: Slice(
                    EnergyRatio(
                        x, chunks=10, with_preconditions=[single_dimension]
                    ),
                    i=i,
                    axis="time",
                ),
                i=i,
            )
            for i in range(10)
        ]
        + [
            partial(
                lambda x, i: Slice(
                    BinnedDistribution(
                        x, bins=10, with_preconditions=[single_dimension]
                    ),
                    i=i,
                    axis="time",
                ),
                i=i,
            )
            for i in range(10)
        ]
        + [
            partial(
                lambda x, i: Slice(
                    BinnedEntropy(
                        x, bins=10, with_preconditions=[single_dimension]
                    ),
                    i=i,
                    axis="time",
                ),
                i=i,
            )
            for i in range(10)
        ]
        + [
            partial(
                lambda x, size, i: Slice(
                    LinearTrend(
                        Aggregate(
                            x,
                            size=size,
                            agg="mean",
                            with_preconditions=[single_dimension],
                        )
                    ),
                    i=i,
                    axis="time",
                ),
                size=size,
                i=i,
            )
            for size in (5, 10, 50)
            for i in range(10)
        ],
        "series-to-attribute-fusion": [],
        "series-to-coefficients-nofusion": [
            lambda x: FFT(x),
            lambda x: CWT(x),
            lambda x: AutoRegressiveCoefficients(x),
            lambda x: FriedrichCoefficients(x),
            lambda x: AutoCorrelation(x),
        ],
        "series-to-coefficients-fusion": [lambda x, y: CrossCorrelation(x, y),],
    }

    if setting == "minimal":
        return minimal
    else:
        return full
