from .autods19 import construct as construct_autods19
from ..computation.util import to_dataframe

__all__ = ['construct']


def construct(X, y, task='classification', transformers='full', return_graph=False, **kwargs):
    """Construct features for a given time series dataset ``X, y``

    Construct features according to the method described in [1]_. The construction method will change in future versions
    of TSFuse.

    Parameters
    ----------
    X : dict
        Feature
    y : numpy.array or pandas.Series
        Target values, given as a 1D array-like object.
    task : {'classification', 'regression'}, optional
        Machine learning task
    transformers : {'minimal', 'fast', 'full'}, optional
        Feature construction settings. The default is `full` which implies using the most
        extensive set of transformers.
    return_graph : bool, optional
        Return the constructed computation graph. By default, only the features are returned.

    Returns
    -------
    features : pandas.DataFrame
        Tabular representation of the constructed features.
    graph : tsfuse.computation.Graph
        Constructed computation graph. Only returned if ``return_graph == True``

    References
    ----------
    .. [1] Arne De Brabandere, Pieter Robberechts, Tim Op De Beéck and Jesse Davis.
       Automating Feature Construction for Multi-View Time Series Data.
       ECML/PKDD Workshop on Automating Data Science 2019.
    """
    graph, data = construct_autods19(
        X,
        y,
        task=task,
        transformers=transformers,
        return_data=True,
        **kwargs,
    )
    df = to_dataframe(data)
    if return_graph:
        return df, graph
    else:
        return df
