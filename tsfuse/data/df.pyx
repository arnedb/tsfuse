import numpy as np

def df_to_arrays_numeric(df):
    # Find windows (assume contiguous)
    _, s, l = np.unique(df['id'].astype(int), return_index=True, return_counts=True)
    cdef long[:] starts = s.astype(long)
    cdef long[:] lengths = l.astype(long)
    cdef double[:, :] df_values = df.iloc[:, 1:].values.astype(float)
    # Create arrays
    cdef double[:, :, :] values = np.full((len(s), max(l), len(df.columns) - 1), np.nan)
    cdef int i
    cdef int N = len(s)
    cdef long start, end
    for i in range(N):
        start = starts[i]
        end = starts[i] + lengths[i]
        values[i, 0:lengths[i], :] = df_values[start:end, :]
    return np.array(values)
