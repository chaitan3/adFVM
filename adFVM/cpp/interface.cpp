#define NO_IMPORT_ARRAY
#include "interface.hpp"

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

PyObject* PyTuple_CreateNone(int n) {
    PyObject* outputs = PyTuple_New(n);
    for (int i = 0; i < n; i++) {
        PyTuple_SetItem(outputs, i, Py_None);
    }
    return outputs;
}

map<string, int> PyOptions_Parse(PyObject* dict) {
    PyObject *key2, *value2;
    map<string, int> options;
    Py_ssize_t pos2 = 0;
    while (PyDict_Next(dict, &pos2, &key2, &value2)) {
        string ckey2 = PyString_AsString(key2);
        options[ckey2] = PyObject_IsTrue(value2);
    }
    return options;
}

extern "C"{
void dsyev_( char* jobz, char* uplo, int* n, double* a, int* lda,
    double* w, double* work, int* lwork, int* info );
void ssyev_( char* jobz, char* uplo, int* n, float* a, int* lda,
    float* w, float* work, int* lwork, int* info );
}

template<typename dtype> void eigenvalue_solver (char* jobz, char* uplo, int* n, dtype* a, int* lda,
dtype* w, dtype* work, int* lwork, int* info );

template<> void eigenvalue_solver (char* jobz, char* uplo, int* n, float* a, int* lda,
float* w, float* work, int* lwork, int* info ) {
    ssyev_(jobz, uplo, n, a, lda, w, work, lwork, info);
}
template<> void eigenvalue_solver (char* jobz, char* uplo, int* n, double* a, int* lda,
double* w, double* work, int* lwork, int* info ) {
    dsyev_(jobz, uplo, n, a, lda, w, work, lwork, info);
}


#ifdef GPU
void Function_get_max_eigenvalue(std::vector<gpuArrType<float, 5, 5>*> phiP) {
    gpuArrType<float, 5, 5>& phi = *phiP[0];
    gpuArrType<float>& eigPhi = *((extArrType<float>*) phiP[1]);
    char jobz = 'N';
    char uplo = 'U';
    int n = 5;
    int lda = 5;
    int lwork = 3*n-1;
    float work[lwork];
    int info;
    float w[5];
    //cout << phi.shape << endl;
    // figure out how to get eigenvalue on GPU
    float* phiData = phi.toHost();
    float* eigPhiData = new float[phi.shape];
    for (int i = 0; i < phi.shape; i++) {
        eigenvalue_solver<float>(&jobz, &uplo, &n, &phiData[25*i], &lda, w, work, &lwork, &info);
        assert(info == 0);
        eigPhiData[i] = w[4];
    }
    gpuErrorCheck(cudaMemcpy(&eigPhi(0), eigPhiData, phi.shape*sizeof(float), cudaMemcpyHostToDevice));
    delete []phiData;
    delete []eigPhiData;
}
#else
void Function_get_max_eigenvalue(std::vector<arrType<scalar, 5, 5>*> phiP) {
    arrType<scalar, 5, 5>& phi = *phiP[0];
    arrType<scalar>& eigPhi = *((arrType<scalar>*) phiP[1]);
    char jobz = 'N';
    char uplo = 'U';
    int n = 5;
    int lda = 5;
    int lwork = 3*n-1;
    scalar work[lwork];
    int info;
    scalar w[5];
    //cout << phi.shape << endl;

    for (int i = 0; i < phi.shape; i++) {
        eigenvalue_solver<scalar>(&jobz, &uplo, &n, &phi(i), &lda, w, work, &lwork, &info);
        assert(info == 0);
        eigPhi(i) = w[4];
    }
}
#endif
