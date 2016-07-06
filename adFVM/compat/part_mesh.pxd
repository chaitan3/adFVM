
cdef extern from "metis.h":
    ctypedef int idx_t
    ctypedef double real_t
    int METIS_PartMeshDual(idx_t *ne, idx_t *nn, idx_t *eptr, idx_t *eind,
                  idx_t *vwgt, idx_t *vsize, idx_t *ncommon, idx_t *nparts, 
                  real_t *tpwgts, idx_t *options, idx_t *objval, idx_t *epart, 
                  idx_t *npart);


