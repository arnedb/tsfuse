import numpy as np
from scipy.stats import norm

from tsfuse.data import Collection


def brownian(N=10, t=100, d=2, random_state=None, **kwargs):
    """
    Generate Brownian motion data.

    The implementation is based on the SciPy Cookbook. [1]_

    Parameters
    ----------
    N : int, optional
        Number of windows. Default: 10
    t : int, optional
        Number of time stamps in each window. Default: 100
    d : int, optional
        Number of dimensions. Default: 2
    random_state : int, optional
        Random state initialization.
    **kwargs
        Keyword arguments to pass to the :class:`~tsfuse.data.Collection` constructor.

    Returns
    -------
    generated : Collection

    References
    ----------
    .. [1] https://scipy-cookbook.readthedocs.io/items/BrownianMotion.html
    """
    if random_state is not None:
        np.random.seed(random_state)

    def generate(n, delta=0.25, dt=0.1, initial=0.0):
        x = np.empty(n)
        x[0] = initial
        for k in range(1, n):
            x[k] = x[k - 1] + norm.rvs(scale=delta ** 2 * dt)
        return x

    values = np.empty((N, t, d), dtype=float)
    for i in range(N):
        for j in range(d):
            values[i, :, j] = generate(t)

    return Collection(values=values, **kwargs)


def series(*collections):
    assert len(set(c.shape for c in collections)) == 1
    assert len(collections) >= 1
    for w in range(collections[0].shape[0]):
        for d in range(collections[0].shape[2]):
            yield ((w, slice(None), d), *[c.values[w, :, d] for c in collections])


def dimensions(*collections):
    assert len(set(c.shape for c in collections)) == 1
    assert len(collections) >= 1
    for w in range(collections[0].shape[0]):
        for s in range(collections[0].shape[1]):
            yield ((w, s, slice(None)), *[c.values[w, s, :] for c in collections])
