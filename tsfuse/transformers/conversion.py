from ..computation import Transformer
from ..data import Collection

from .util import mask_nan


class SIUnits(Transformer):

    @staticmethod
    def apply(x):
        if x.unit is None:
            return x
        else:
            si_unit = x.unit.to_base_units()
            if x.unit == si_unit:
                return x
            else:
                values = mask_nan(x) * x.unit
                return Collection.from_array(
                    values.to(si_unit).values,
                    time=x.time,
                    dims=x.dims
                )

    @staticmethod
    def unit(x):
        return x.unit.to_base_units()
