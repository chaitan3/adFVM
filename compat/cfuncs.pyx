cimport numpy as np

ctypedef double dtype

def add_at(np.ndarray[dtype, ndim=2] a, np.ndarray[int] indices, np.ndarray[dtype,ndim=2] b):
    cdef int n = indices.shape[0]
    cdef int m = a.shape[1]
    for i in range(0, n):
        for j in range(0, m):
            a[indices[i], j] += b[i, j]
