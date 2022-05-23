import pandas as pd

from .algorithm import TSFuseExtractor

__all__ = ["construct"]


def construct(
    X,
    y,
    task="auto",
    transformers="full",
    max_depth=2,
    return_graph=False,
    **kwargs
):
    """
    Construct features for a labeled time series dataset ``X, y``

    Parameters
    ----------
    X : dict(str, Collection)
        Time series data.
    y : array-like
        Labels. Since each window has a single label, the length should be
        equal to the number of windows in the given time series data.
    task : {'classification', 'regression', 'auto'}, default: 'auto'
        Machine learning task: detected automatically by default.
    transformers : {'minimal', 'full'}, default: 'full'
        Feature construction settings: 'minimal' uses a minimal set of
        simple statistical transformers and 'full' the complete set of
        transformers.
    max_depth : int, default: 2
        Maximum computation graph depth.
    return_graph : bool, default: False
        Return computation graph.

    Returns
    -------
    features : pandas.DataFrame
        Constructed features.
    graph : Graph
        Computation graph that computes the constructed features.
        Only returned if ``return_graph == True``
    """
    extractor = TSFuseExtractor(
        transformers=transformers, task=task, max_depth=max_depth, **kwargs
    )
    features = extractor.fit_transform(X, pd.Series(y))
    if return_graph:
        return features, extractor.graph_
    else:
        return features
