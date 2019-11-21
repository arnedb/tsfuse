__all__ = [
    'aggregate',
]

import numpy as np

cimport cython
cimport numpy as np

cdef extern from "<math.h>" nogil:
    const float NAN
    const float INFINITY
    bint isnan(double x)
    double sqrt(double x)
    double log(double x)
    double pow(double x, double y)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def aggregate(values, int size, int agg):
    # Input array
    if not np.issubdtype(values.dtype, np.float64):
        raise AttributeError()
    cdef double[:, :, :] x = values

    # Create results array
    cdef int chunks = np.ceil(float(x.shape[2]) / size)
    shape = (x.shape[0], x.shape[1], chunks)
    cdef double[:, :, :] result = np.empty(shape, dtype=np.float64)

    # Variable definitions
    cdef int i, j, l, c, count, start, stop
    cdef double val, n1, n, delta, delta_n, delta_n2, term1, mean, m2, m3, m4, min, max, var, sum

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for c in range(chunks):
                start = c * size
                stop = (c + 1) * size
                if stop > x.shape[2]:
                    stop = x.shape[2]
                min = INFINITY
                max = -INFINITY
                n1 = 0
                n = 0
                delta = 0
                delta_n = 0
                delta_n2 = 0
                term1 = 0
                mean = 0
                m2 = 0
                count = 0
                sum = 0
                for l in range(start, stop):
                    val = x[i, j, l]
                    if (isnan(val) == 0) and (val != INFINITY) and (val != -INFINITY):
                        # Update count
                        count = count + 1
                        # Update sum
                        sum = sum + val
                        # Update min
                        if val < min:
                            min = val
                        # Update max
                        if val > max:
                            max = val
                        # Update moments
                        n1 = n
                        n = n + 1.0
                        delta = val - mean
                        delta_n = delta / n
                        delta_n2 = delta_n * delta_n
                        term1 = delta * delta_n * n1
                        mean = mean + delta_n
                        m2 = m2 + term1
                if count > 0:
                    var = m2 / count
                else:
                    min = NAN
                    max = NAN
                    mean = NAN
                    var = NAN
                if agg == 0:
                    result[i, j, c] = mean
                if agg == 1:
                    result[i, j, c] = var
                if agg == 2:
                    result[i, j, c] = min
                if agg == 3:
                    result[i, j, c] = max

    return np.array(result)
