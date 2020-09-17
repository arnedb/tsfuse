from .autods19 import construct as construct_autods19
from .unsupervised import construct as construct_unsupervised

__all__ = ['construct']


def construct(X, y=None, task='classification', transformers='full', return_graph=False, **kwargs):
    """
    Construct features for a given time series dataset ``X, y``

    This function implements the method presented in our paper in the ECML/PKDD Workshop on
    Automating Data Science 2019. [1]_ The construction method will change in future versions of TSFuse.

    Parameters
    ----------
    X : dict(str, Collection)
        Multi-view time series data.
    y : array-like, optional
        Target data. Not required for unsupervised feature construction.
    task : {'classification', 'regression'}, optional
        Machine learning task. Default: `classification`
    transformers : {'minimal', 'fast', 'full'}, optional
        Feature construction settings. Default: `full` (the most extensive set of transformers)
    return_graph : bool, optional
        Return the computation graph. Default: `False`

    Returns
    -------
    features : pandas.DataFrame
        Tabular representation of the constructed features.
    graph : Graph
        Constructed computation graph. Only returned if ``return_graph == True``

    References
    ----------
    .. [1] Arne De Brabandere, Pieter Robberechts, Tim Op De Beéck and Jesse Davis.
       `Automating Feature Construction for Multi-View Time Series Data <https://www.google.com/url?q=https%3A%2F%2Fupvedues-my.sharepoint.com%2F%3Ab%3A%2Fg%2Fpersonal%2Fjorallo_upv_edu_es%2FETxycG2WhmFBmVN7CNW8yKsBQHwhhlzdyegEx1AnNeRa2w%3Fe%3DbPQR7e&sa=D&sntz=1&usg=AFQjCNH-zTIQtPE2M0m0h_uUPN_25SaGCw>`_.
       ECML/PKDD Workshop on Automating Data Science 2019.
    """
    if y is not None:
        graph, data = construct_autods19(
            X,
            y,
            task=task,
            transformers=transformers,
            return_data=True,
            **kwargs,
        )
    else:
        graph, data = construct_unsupervised(
            X,
            transformers=transformers,
            return_data=True,
            **kwargs,
        )
    if return_graph:
        return data, graph
    else:
        return data
