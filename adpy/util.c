#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>

typedef double scalar;
typedef int32_t integer;

typedef struct {
    char * data;
    int64_t *shape;
    int dims;
    int64_t size;
    int type;
    int bytes;
} ndarray;

ndarray* ndarray_from_numpy(PyArrayObject *arr) {
    ndarray *ndarr = malloc(sizeof(ndarray));
    ndarr -> data = arr -> data;
    ndarr -> dims = arr -> nd;
    ndarr -> shape = arr -> dimensions;
    ndarr -> size = 1;
    ndarr -> type = arr -> descr -> type_num;
    ndarr -> bytes = arr -> descr -> elsize;
    for (int64_t i = 0; i < ndarr -> dims; i++) {
        ndarr -> size *= ndarr -> shape[i];
    }
    //printf("%d %f\n", ndarr->size, ((double*)ndarr->data)[0]);
    return ndarr;
}

ndarray* ndarray_alloc(ndarray* ndarr_ref) {
    ndarray *ndarr = malloc(sizeof(ndarray));
    memcpy(ndarr, ndarr_ref, sizeof(ndarray));

    int shape_size = sizeof(int64_t)*ndarr->dims;
    ndarr -> shape = malloc(shape_size);
    memcpy(ndarr->shape, ndarr_ref->shape, shape_size);

    int64_t data_size = ndarr -> size * ndarr -> bytes;
    ndarr -> data = malloc(data_size);
    memset(ndarr->data, 0, data_size);

    return ndarr;
}

//ndarray* ndarray_build(char* data, int64_t* shape, int dims)
//    ndarray *ndarr = malloc(sizeof(ndarray));
//    ndarr -> data = data;
//    ndarr -> shape = shape;
//    ndarr -> dims = dims;
//    ndarr -> size = 1;
//    for (int64_t i = 0; i < ndarr -> dims; i++) {
//        ndarr -> size *= ndarr -> shape[i];
//    }
//    ndarr -> type = NPY_DOUBLE;
//    ndarr -> bytes = 8;
//    return ndarr;
//}


void ndarray_free(ndarray* ndarr) {
    free(ndarr->data);
    free(ndarr->shape);
    free(ndarr);
}

ndarray* ndarray_multiply(ndarray *ndarr, scalar constant) {
    ndarray* ndarr2 = ndarray_alloc(ndarr);
    scalar *data0 = ndarr -> data;
    scalar *data1 = ndarr2 -> data;
    int64_t i;
    for (i = 0; i < ndarr -> size; i++) {
        data1[i] = data0[i]*constant;
    }
    //printf("data1[0] %f\n", ((scalar *)ndarr2->data)[0]);
    return ndarr2;
}

PyArrayObject* numpy_from_ndarray(ndarray *ndarr) {
    PyArrayObject* arr = PyArray_SimpleNewFromData(ndarr->dims, ndarr->shape, ndarr->type, ndarr->data);
    PyArray_ENABLEFLAGS(arr, NPY_ARRAY_OWNDATA);
    return arr;
}

PyObject* test_interface(PyObject* self, PyObject *args) {
    PyArrayObject *input;
    if (!PyArg_ParseTuple(args, "O", &input)) {
        return NULL;
    }
    ndarray *ndarr = ndarray_from_numpy(input);

    ndarray* ndarr2 = ndarray_multiply(&ndarr, 3.84);
    //ndarray_free(ndarr2);

    PyArrayObject* output = numpy_from_ndarray(ndarr2);
    return (PyObject*)output;
}

#include "interface.c"

static PyMethodDef methods[] =
{
     {"test_interface", test_interface, METH_VARARGS, "wtf"},
 #ifdef FUNCTION_INTERFACE
     {"interface", interface, METH_VARARGS, "wtf"},
 #endif
     {NULL, NULL, 0, NULL}
};

/* module initialization */
PyMODINIT_FUNC
initfunction(void)
{
     (void) Py_InitModule("function", methods);
     /* IMPORTANT: this must be called */
     import_array();
}

//int main(int argc, char**argv) {
//    return 0;
//}


