import numpy as np

cimport numpy as cnp

def df_to_arrays_numeric(df):
    # Find windows (assume contiguous)
    _, s, l = np.unique(df['id'].astype(int), return_index=True, return_counts=True)
    cdef cnp.intp_t[:] starts = s.astype(np.intp)
    cdef cnp.intp_t[:] lengths = l.astype(np.intp)
    cdef double[:, :] df_values = df.iloc[:, 1:].values.astype(float)
    # Create arrays
    cdef double[:, :, :] values = np.full((len(s), max(l), len(df.columns) - 1), np.nan)
    cdef int i
    cdef int N = len(s)
    cdef cnp.intp_t start, end
    for i in range(N):
        start = starts[i]
        end = starts[i] + lengths[i]
        values[i, 0:lengths[i], :] = df_values[start:end, :]
    return np.array(values)
