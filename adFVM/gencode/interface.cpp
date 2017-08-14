#include "interface.hpp"
#include "mesh.hpp"

long long mil = 0;
struct memory mem = {0, 0};

long long current_timestamp() {
    struct timeval te; 
    gettimeofday(&te, NULL); // get current time
    long long milliseconds = te.tv_sec*1000LL + te.tv_usec/1000; // caculate milliseconds
    //printf("milliseconds: %lld\n", milliseconds);
    return milliseconds;
}

void Mesh::build() {}
void Mesh::buildBeforeWrite() {}

Mesh *meshp = NULL;
#ifdef MATOP
    #include "matop.hpp"
    Matop *matop;
#endif

#define MODULE interface
#ifdef PY3
    #define initFunc GET_MODULE(PyInit_,MODULE)
#else
    #define initFunc GET_MODULE(init,MODULE)
#endif
#define modName VALUE(MODULE)

PyObject* initSolver(PyObject *self, PyObject *args) {

    PyObject *meshObject = PyTuple_GetItem(args, 0);
    Py_INCREF(meshObject);

    meshp = new Mesh(meshObject);
    meshp->init();

    #ifdef GPU
        int count;
        gpuErrorCheck(cudaGetDeviceCount(&count));
        printf("GPU devices: %d, rank: %d\n", count, meshp->rank);
    #endif

    #ifdef MATOP
        integer argc = 0;
        PetscInitialize(&argc, NULL, NULL, NULL);
        matop = new Matop(rcf);
    #endif

    return Py_None;
}

PyObject* viscosity(PyObject *self, PyObject *args) {

    //cout << "forward 1" << endl;
    PyObject *uObject, *rhoObject, *rhoUObject, *rhoEObject;
    bool report;
    scalar dt, scaling;
    PyArg_ParseTuple(args, "OOOOddb", &uObject, &rhoObject, &rhoUObject, &rhoEObject, &scaling, &dt, &report);

    arrType<scalar, 5> u;
    vec rho, rhoE;
    mat rhoU;
    getArray((PyArrayObject *)uObject, u);
    getArray((PyArrayObject *)rhoObject, rho);
    getArray((PyArrayObject *)rhoUObject, rhoU);
    getArray((PyArrayObject *)rhoEObject, rhoE);
    const Mesh& mesh = *meshp;

    vec M_2norm(mesh.nCells);
    vec DT(mesh.nFaces);

    //matop->viscosity(rho, rhoU, rhoE, M_2norm, DT, scaling, report);
    //return putArray(M_2norm);

    arrType<scalar, 5> un(mesh.nInternalCells);
    #ifdef MATOP
        matop->viscosity(rho, rhoU, rhoE, M_2norm, DT, scaling, report);
        matop->heat_equation(rcf, u, DT, dt, un);
    #endif
    
    return putArray(un);
}

PyObject* finalSolver(PyObject *self, PyObject *args) {
    //delete rcf;
    delete meshp;
    #ifdef MATOP
        PetscFinalize();
        delete matop;
    #endif
    return NULL;
}

extern PyMethodDef Methods[];

#ifdef PY3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,  /* m_base */
    modName,                 /* m_name */
    NULL,                   /* m_doc */
    -1,                     /* m_size */
    Methods            /* m_methods */
};
#endif

PyMODINIT_FUNC
initFunc(void)
{
    PyObject *m;

    #ifdef PY3
        m = PyModule_Create(&moduledef);
    #else
        m = Py_InitModule(modName, Methods);
        if (m == NULL)
            return;
    #endif
    import_array();

    //SpamError = PyErr_NewException("spam.error", NULL, NULL);
    //Py_INCREF(SpamError);
    //PyModule_AddObject(m, "error", SpamError);
    #ifdef PY3
        return m;
    #endif
}


#ifdef PY3
char* PyString_AsString(PyObject* result) {
    char *my_result;
    PyObject * temp_bytes = PyUnicode_AsEncodedString(result, "ASCII", "strict"); // Owned reference
    if (temp_bytes != NULL) {
        my_result = PyBytes_AS_STRING(temp_bytes); // Borrowed pointer
        my_result = strdup(my_result);
        Py_DECREF(temp_bytes);
        return my_result;
    } else {
        return NULL;
    }
}
#endif

#ifdef MATOP
    extern "C"{
    void dsyev_( char* jobz, char* uplo, int* n, double* a, int* lda,
        double* w, double* work, int* lwork, int* info );
    }

    scalar getMaxEigenvalue(arrType<double, 5, 5>& phi, vec& eigPhi) {
        char jobz = 'N';
        char uplo = 'U';
        int n = 5;
        int lda = 5;
        int lwork = 3*n-1;
        double work[lwork];
        int info;
        double w[5];

        for (int i = 0; i < phi.shape; i++) {
            dsyev_(&jobz, &uplo, &n, &phi(i), &lda, w, work, &lwork, &info);
            assert(info == 0);
            eigPhi(i) = w[4];
        }
        return 0;
    }
#endif


