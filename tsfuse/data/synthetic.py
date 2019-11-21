import numpy as np
from scipy.stats import norm

from tsfuse.data import Collection


def brownian(n_windows=10, n_timestamps=100, n_dimensions=2):
    # Reference: https://scipy-cookbook.readthedocs.io/items/BrownianMotion.html
    def generate(n, delta=0.25, dt=0.1, initial=0.0):
        x = np.empty(n)
        x[0] = initial
        for k in range(1, n):
            x[k] = x[k - 1] + norm.rvs(scale=delta ** 2 * dt)
        return x

    values = np.empty((n_windows, n_timestamps, n_dimensions))
    for w in range(n_windows):
        for d in range(n_dimensions):
            values[w, :, d] = generate(n_timestamps)

    return Collection(values)


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