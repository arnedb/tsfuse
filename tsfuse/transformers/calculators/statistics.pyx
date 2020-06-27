__all__ = [
    'single_pass_statistics',
    'longest_non_zero_strike',
    'auto_correlation',
    'cross_correlation',
    'binned_distribution',
    'entropy',
    'sample_entropy',
    'energy_ratio',
    'spectral_moment',
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
    double fabs(double x)
    double pow(double x, double y)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def single_pass_statistics(values):
    # Input array
    if not np.issubdtype(values.dtype, np.float64):
        raise AttributeError()
    cdef double[:, :, :] x = values
    # Create results array
    shape = (x.shape[0], x.shape[1], 8)
    cdef double[:, :, :] result = np.empty(shape, dtype=np.float64)

    # Variable definitions
    cdef int i, j, l, count
    cdef double val, n1, n, delta, delta_n, delta_n2, term1, mean, m2, m3, m4

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            result[i, j, 0] = 0
            result[i, j, 1] = 0
            result[i, j, 2] = INFINITY
            result[i, j, 3] = -INFINITY
            n1 = 0
            n = 0
            delta = 0
            delta_n = 0
            delta_n2 = 0
            term1 = 0
            mean = 0
            m2 = 0
            m3 = 0
            m4 = 0
            for l in range(x.shape[2]):
                val = x[i, j, l]
                if (isnan(val) == 0) and (val != INFINITY) and (val != -INFINITY):
                    # Update count
                    result[i, j, 0] = result[i, j, 0] + 1
                    # Update sum
                    result[i, j, 1] = result[i, j, 1]  + val
                    # Update min
                    if val < result[i, j, 2]:
                        result[i, j, 2] = val
                    # Update max
                    if val > result[i, j, 3]:
                        result[i, j, 3] = val
                    # Update moments
                    n1 = n
                    n = n + 1.0
                    delta = val - mean
                    delta_n = delta / n
                    delta_n2 = delta_n * delta_n
                    term1 = delta * delta_n * n1
                    mean = mean + delta_n
                    m4 = m4 + term1 * delta_n2 * (n*n - 3*n + 3) + 6 * delta_n2 * m2 - 4 * delta_n * m3
                    m3 = m3 + term1 * delta_n * (n - 2) - 3 * delta_n * m2
                    m2 = m2 + term1
            if result[i, j, 0] > 0:
                # Compute mean, variance, skewness and kurtosis
                result[i, j, 4] = mean
                result[i, j, 5] = m2 / result[i, j, 0]
                if m2 != 0:
                    result[i, j, 6] = (sqrt(n) * m3) / pow(m2, 1.5)
                    result[i, j, 7] = (n * m4) / (m2 * m2) - 3
                else:
                    result[i, j, 6] = NAN
                    result[i, j, 7] = NAN
            else:
                result[i, j, 2] = NAN
                result[i, j, 3] = NAN
                result[i, j, 4] = NAN
                result[i, j, 5] = NAN
                result[i, j, 6] = NAN
                result[i, j, 7] = NAN

    return np.array(result)

@cython.boundscheck(False)
@cython.wraparound(False)
def longest_non_zero_strike(values):
    # Input array
    if np.issubdtype(values.dtype, np.bool_):
        values = values.astype(np.float64)
    if not np.issubdtype(values.dtype, np.float64):
        raise AttributeError()
    cdef double[:, :, :] x = values
    # Create results array
    shape = (x.shape[0], x.shape[1], 1)
    cdef double[:, :, :] result = np.empty(shape, dtype=np.float64)

    # Variable definitions
    cdef int i, j, k
    cdef int longest_strike = 0
    cdef int current_strike = 0

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            longest_strike = 0
            current_strike = 0
            for k in range(x.shape[2]):
                if x[i, j, k] != 0:
                    current_strike += 1
                    if current_strike > longest_strike:
                        longest_strike = current_strike
                else:
                    current_strike = 0
            result[i, j, 0] = longest_strike

    return np.array(result)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def auto_correlation(values):
    # Input array
    if not np.issubdtype(values.dtype, np.float64):
        raise AttributeError()
    cdef double[:, :, :] x = values

    # Create results array
    shape = (x.shape[0], x.shape[1], x.shape[2])
    cdef double[:, :, :] result = np.empty(shape, dtype=np.float64)

    # Mean and variance
    cdef double[:, :] mu = np.nanmean(x, axis=2)
    cdef double[:, :] sig2 = np.nanvar(x, axis=2)

     # Variable definitions
    cdef int i, j, lag, t
    cdef double sum

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for lag in range(x.shape[2]):
                sum = 0.0
                for t in range(x.shape[2] - lag):
                    sum += (x[i, j, t] - mu[i, j]) * (x[i, j, t + lag] - mu[i, j])
                result[i, j, lag] = 1 / ((x.shape[2] - lag) * sig2[i, j]) * sum

    return np.array(result)


@cython.boundscheck(False)
@cython.wraparound(False)
def cross_correlation(values1, values2):
    # Input arrays
    if not np.issubdtype(values1.dtype, np.float64):
        raise AttributeError()
    if not np.issubdtype(values2.dtype, np.float64):
        raise AttributeError()
    if values1.shape != values2.shape:
        raise AttributeError()
    cdef double[:, :, :] x = values1
    cdef double[:, :, :] y = values2

    # Create results array
    shape = (x.shape[0], x.shape[1], x.shape[2])
    cdef double[:, :, :] result = np.empty(shape, dtype=np.float64)

     # Variable definitions
    cdef int i, j, k, lag

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for lag in range(x.shape[2]):
                result[i, j, lag] = 0
                for k in range(x.shape[2] - lag):
                    result[i, j, lag] += x[i, j, lag + k] * y[i, j, k]

    return np.array(result)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def binned_distribution(values, bins=10):
    # Input array
    if not np.issubdtype(values.dtype, np.float64):
        raise AttributeError()
    cdef double[:, :, :] x = values

    # Create results array
    shape = (x.shape[0], x.shape[1], bins)
    cdef double[:, :, :] result = np.zeros(shape, dtype=np.float64)

    # Variable definitions
    cdef int i, j, l, u
    cdef int bin
    cdef double val, min, max, binsize, lower, upper, total
    cdef int n = bins

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            # Find min and max
            min = INFINITY
            max = -INFINITY
            for l in range(x.shape[2]):
                val = x[i, j, l]
                if (isnan(val) == 0) and (val != INFINITY) and (val != -INFINITY):
                    if val < min:
                        min = val
                    if val > max:
                        max = val
            # Stop if interval is zero or infinite
            if (min == max) or (min == INFINITY) or (max == -INFINITY):
                 break
            binsize = (max - min) / n
            # Count values per bin
            total = 0
            for l in range(x.shape[2]):
                val = x[i, j, l]
                if (isnan(val) == 0):
                    for bin in range(n):
                        lower = min + bin * binsize
                        upper = min + (bin + 1) * binsize
                        if bin < n - 1:
                            if (val >= lower) and (val < upper):
                                result[i, j, bin] = result[i, j, bin] + 1
                                total = total + 1
                        else:
                            # Last bin: include upper bound
                            if (val >= lower):
                                result[i, j, bin] = result[i, j, bin] + 1
                                total = total + 1
            # Normalize counts
            if total > 0:
                for bin in range(n):
                    result[i, j, bin] = result[i, j, bin] / total

    return np.array(result)


@cython.boundscheck(False)
@cython.wraparound(False)
def entropy(values):
    # Input array
    if not np.issubdtype(values.dtype, np.float64):
        raise AttributeError()

    # Normalize
    values = values / np.sum(values, axis=2, keepdims=True)

    cdef double[:, :, :] x = values

    cdef int i, j, l

    # Create results array
    shape = (x.shape[0], x.shape[1], 1)
    cdef double[:, :, :] result = np.full(shape, dtype=np.float64, fill_value=np.nan)


    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
                result[i, j, 0] = 0.0
                for l in range(x.shape[2]):
                    if x[i, j, l] != 0:
                        result[i, j, 0] += - x[i, j, l] * log(x[i, j, l])

    return np.array(result)

@cython.boundscheck(False)
@cython.wraparound(False)
def sample_entropy(values):
    # Input array
    if not np.issubdtype(values.dtype, np.float64):
        raise AttributeError()

    cdef double[:, :, :] x = values

    cdef int M = 2
    cdef double[:, :] tolerance = 0.2 * np.nanstd(values, axis=2)
    cdef int n = x.shape[2]
    cdef double[:] v = np.full((n,), dtype=np.float64, fill_value=np.nan)

    cdef int i, j, l

    # Create results array
    shape = (x.shape[0], x.shape[1], 1)
    cdef double[:, :, :] result = np.full(shape, dtype=np.float64, fill_value=np.nan)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
                # Copy as one array
                for l in range(x.shape[2]):
                    v[l] = x[i, j, l]
                # Compute sample entropy
                result[i, j, 0] = sampen(v, M, tolerance[i, j])

    return np.array(result)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double sampen(double[:] L, int m, double r):
    cdef int N = L.shape[0]
    cdef double B = 0.0
    cdef double A = 0.0
    cdef int i, j, k
    cdef double dist
    cdef double d

    # Compute B
    for i in range(N - m):
        for j in range(N - m):
            if j > i:
                dist = 0.0
                for k in range(m):
                    d = fabs(L[i+k]-L[j+k])
                    if d > dist:
                        dist = d
                if dist < r:
                    B = B + 1

    # Compute A
    for i in range(N - m):
        for j in range(N - m):
            if j > i:
                dist = 0.0
                for k in range(m + 1):
                    d = fabs(L[i+k]-L[j+k])
                    if d > dist:
                        dist = d
                if dist < r:
                    A = A + 1

    # Return SampEn
    return - log(A / B)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def energy_ratio(values, int chunks):
    # Input array
    if not np.issubdtype(values.dtype, np.float64):
        raise AttributeError()

    cdef double[:, :, :] x = values

    cdef int i, j, l, t, c

    cdef int chunks1 = x.shape[2] % chunks
    cdef int length1 = x.shape[2] // chunks + 1
    cdef int length2 = x.shape[2] // chunks

    cdef double total = 0.0

    shape = (x.shape[0], x.shape[1], chunks)
    cdef double[:, :, :] result = np.full(shape, dtype=np.float64, fill_value=0)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
                l = 0
                for c in range(chunks1):
                    for t in range(length1):
                        result[i, j, c] += x[i, j, l] * x[i, j, l]
                        l += 1
                    total += result[i, j, c]
                for c in range(chunks1, chunks):
                    for t in range(length2):
                        result[i, j, c] += x[i, j, l] * x[i, j, l]
                        l += 1
                    total += result[i, j, c]

                if total > 0:
                    for c in range(chunks):
                        result[i, j, c] = result[i, j, c] / total

    return np.array(result)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def spectral_moment(values, int r, bint origin):
    # Input array
    if not np.issubdtype(values.dtype, np.float64):
        raise AttributeError()

    cdef double[:, :, :] x = values

    cdef int i, j, l
    cdef double sum

    shape = (x.shape[0], x.shape[1], 1)
    cdef double[:, :, :] result = np.full(shape, dtype=np.float64, fill_value=np.nan)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):

                result[i, j, 0] = 0.0
                sum = 0.0
                for l in range(x.shape[2]):
                    result[i, j, 0] += pow(l, r) * x[i, j, l]
                    sum += x[i, j, l]

                result[i, j, 0] = result[i, j, 0] / sum

    return np.array(result)
