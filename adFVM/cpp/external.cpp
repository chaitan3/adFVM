#define NO_IMPORT_ARRAY
#include "external.hpp"

void Mesh::build() {}
void Mesh::buildBeforeWrite() {}

Mesh *meshp = NULL;
#ifdef GPU
    cusolverDnHandle_t cusolver_handle;
    cublasHandle_t cublas_handle;
#endif
#ifdef MATOP
    #include "matop.hpp"
    Matop *matop;
#endif

void external_init (PyObject* args) {
    int rank = PyInt_AsLong(PyTuple_GetItem(args, 0));
    PyObject *meshObject = PyTuple_GetItem(args, 1);
    Py_INCREF(meshObject);

    meshp = new Mesh(meshObject);
    meshp->init();
    meshp->localRank = rank;

    parallel_init();

    #ifdef GPU
        auto status1 = cusolverDnCreate(&cusolver_handle);
        assert(status1 == CUSOLVER_STATUS_SUCCESS);
        auto status2 = cublasCreate(&cublas_handle);
        assert(status2 == CUBLAS_STATUS_SUCCESS);
    #endif

    #ifdef MATOP
        matop = new Matop();
    #endif
}

void external_exit () {
    parallel_exit();
    delete meshp;
    #ifdef MATOP
        delete matop;
    #endif
}
