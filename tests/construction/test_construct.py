import numpy as np
import pandas as pd

from tsfuse import construct
from tsfuse.data.synthetic import brownian


def test_construct():
    x1, x2, x3 = (
        brownian(N=10, random_state=0),
        brownian(N=10, random_state=1),
        brownian(N=10, random_state=2),
    )
    X = {"x1": x1, "x2": x2, "x3": x3}
    y = pd.Series(
        np.mean(x1.values[:, :, 0], axis=1)
        / np.mean(x3.values[:, :, 0], axis=1)
    )
    features, graph = construct(X, y, transformers="minimal", return_graph=True)
    assert features.shape[1] > 0
    assert len(graph.outputs) > 0
    assert features.shape[1] == len(graph.outputs)

