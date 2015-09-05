cimport numpy as np

ctypedef double dtype

def add_at(np.ndarray[dtype, ndim=2] a, np.ndarray[int] indices, np.ndarray[dtype,ndim=2] b):
    cdef int n = indices.shape[0]
    cdef int m = a.shape[1]
    for i in range(0, n):
        for j in range(0, m):
            a[indices[i], j] += b[i, j]

def intersect(np.ndarray[int, ndim=2] faces, np.ndarray[dtype, ndim=2] points, np.ndarray[dtype] point, np.ndarray[dtype] normal):
    cdef int n = faces.shape[0]
    cdef int d = faces.shape[1]
    left = np.zeros(d, dtype=int)
    right = np.zeros(d, dtype=int)

    for i in range(0, n):
        for j in range(0, d):
            points[faces[i, j]]
        

        
