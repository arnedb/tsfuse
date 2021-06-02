from copy import deepcopy

from .graph import *
from .nodes import *

__all__ = [
    'extract',
    'Graph',
    'Node',
    'Input',
    'Constant',
    'Transformer',
]

def extract(X, transformers='full', return_graph=False):
    """
    Compute a given list of series-to-attribute transformers
    for given time series data ``X``

    Parameters
    ----------
    X : dict(str, Collection)
        Multi-view time series data.
    transformers : {'minimal', 'fast', 'full'} or list, optional
        List of transformers: 'minimal', 'fast' or 'full' (default).
        Alternatively, it is possible to give an explicit list of transformers.
    return_graph : bool, optional
        Return the computation graph. Default: `False`

    Returns
    -------
    features : pandas.DataFrame
        Tabular representation of the constructed features.
    graph : Graph
        Constructed computation graph. Only returned if ``return_graph == True``
    """
    from tsfuse.construction.autods19 import minimal, fast, full
            
    if transformers == 'minimal':
        transformers = minimal['series-to-attribute']
    elif transformers == 'fast':
        transformers = fast['series-to-attribute']
    elif transformers == 'full':
        transformers = full['series-to-attribute']
    else:
        transformers = deepcopy(transformers)
    
    graph = Graph()
    for v in X:
        i = Input(v)
        for t in transformers:
            transformer = deepcopy(t)
            transformer._parents = [i]
            transformer._is_output = True
            graph.add_node(transformer)

    features = graph.transform(X, return_dataframe=True)

    if return_graph:
        return features, graph
    else:
        return features