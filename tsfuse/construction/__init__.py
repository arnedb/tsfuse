import pandas as pd

from .algorithm import TSFuseExtractor

__all__ = ["construct"]


def construct(
    X, y, task="auto", transformers="full", return_graph=False, **kwargs
):
    """
    Construct features for a given time series dataset ``X, y``

    Parameters
    ----------
    X : dict(str, Collection)
        Multi-view time series data.
    y : array-like
        Target data.
    task : {'classification', 'regression', 'auto'}, optional
        Machine learning task. Default: `auto` (detect task automatically)
    transformers : {'minimal', 'fast', 'full'}, optional
        Feature construction settings.
        Default: `full` (the most extensive set of transformers)
    return_graph : bool, optional
        Return the computation graph. Default: `False`

    Returns
    -------
    features : pandas.DataFrame
        Tabular representation of the constructed features.
    graph : Graph
        Constructed computation graph.
        Only returned if ``return_graph == True``
    """
    extractor = TSFuseExtractor(transformers=transformers, task=task, **kwargs)
    features = extractor.fit_transform(X, pd.Series(y))
    if return_graph:
        return features, extractor.graph_
    else:
        return features
