import numpy as np

from tsfuse.data.units import units
from ..computation import Transformer, Graph
from .mathematics import Sum, ArcCos
from .util import apply_to_axis

__all__ = [
    'Norm',
    'Resultant',
    'Angle',
]


class Norm(Transformer):
    """
    Vector norm

    Parameters
    ----------
    p : int, optional
        Order of the vector norm. Default: 2
    """
    def __init__(self, *parents, p=2, **kwargs):
        super(Norm, self).__init__(*parents, **kwargs)
        self.p = p
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
            lambda x: x.shape[2] > 1,
        ]

    def transform(self, x, **kwargs):
        """
        Compute the vector norm of order :math:`p` over the dimensions of **x**

        Parameters
        ----------
        x : Collection
            Multivariate time series data with least 2 dimensions.
        """
        return super().transform(x, **kwargs)

    def apply(self, x):
        def calculator(a):
            return np.linalg.norm(a, ord=self.p, axis=-1, keepdims=True)

        return apply_to_axis(calculator, x, axis='dims')


class Resultant(Transformer):
    """
    Euclidean norm
    """
    def __init__(self, *parents, **kwargs):
        super(Resultant, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 1,
            lambda x: np.issubdtype(x.dtype, np.float64),
            lambda x: x.shape[2] > 1,
        ]

    def transform(self, x, **kwargs):
        """
        Compute the Euclidean norm over the dimensions of **x**

        Parameters
        ----------
        x : Collection
            Multivariate time series data with least 2 dimensions.
        """
        return super().transform(x, **kwargs)

    def graph(self, x):
        return Graph(Norm(x, p=2))


class Angle(Transformer):
    """
    Angle defined by three points
    """
    def __init__(self, *parents, **kwargs):
        super(Angle, self).__init__(*parents, **kwargs)
        self.preconditions = [
            lambda *collections: len(collections) == 3,
            lambda *collections: all(c.shape[2] > 1 for c in collections),
            lambda p1, p2, p3: np.issubdtype(p1.dtype, np.float64),
            lambda p1, p2, p3: np.issubdtype(p2.dtype, np.float64),
            lambda p1, p2, p3: np.issubdtype(p3.dtype, np.float64),
            lambda p1, p2, p3: len(set(p.shape for p in (p1, p2, p3))) == 1,
        ]

    def transform(self, p1, p2, p3, **kwargs):
        """
        Compute the angle :math:`\mathbf{\\theta}` defined by three points
        **p1**, **p2**, **p3** as shown in the figure below:

        .. image:: angles.png
           :width: 200px
           :align: center

        This function uses the following formula for computing :math:`\\theta`:

        .. centered::
           :math:`\\theta = \mathrm{arccos}\\Bigg( \\frac{\overrightarrow{p1p2}~\cdot~\overrightarrow{p2p3}}{||\overrightarrow{p1p2}||~||\overrightarrow{p2p3}||} \\Bigg)`
        
        Parameters
        ----------
        p1 : Collection
            2D/3D coordinates of point 1.
        p2 : Collection
            2D/3D coordinates of point 2.
        p3 : Collection
            2D/3D coordinates of point 3.
        """
        return super().transform(p1, p2, p3, **kwargs)

    def graph(self, p1, p2, p3):
        v1 = p2 - p1
        l1 = Norm(v1)
        v2 = p3 - p2
        l2 = Norm(v2)
        dot = Sum(v1 * v2, axis='dims')
        theta = ArcCos(dot / (l1 * l2))
        return Graph(theta)
