#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>

typedef double scalar;
typedef int32_t integer;

typedef struct {
    char * data;
    int type;
    int bytes;
    int dims;
    int64_t *shape;
    int64_t size;
} ndarray;

void ndarray_from_numpy(ndarray *ndarr, PyArrayObject *arr) {
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
}

ndarray* ndarray_alloc(ndarray* ndarr_ref) {
    ndarray *ndarr = malloc(sizeof(ndarray));
    memcpy(ndarr, ndarr_ref, sizeof(ndarray));

    int shape_size = sizeof(int)*ndarr->dims;
    ndarr -> shape = malloc(shape_size);
    memcpy(ndarr->shape, ndarr_ref->shape, shape_size);

    int64_t data_size = ndarr -> size * ndarr -> bytes;
    ndarr -> data = malloc(data_size);
    memset(ndarr->shape, 0, data_size);

    return ndarr;
}

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
    PyArrayObject* arr = PyArray_SimpleNewFromData(ndarr->dims, ndarr->shape, NPY_DOUBLE, ndarr->data);
    return arr;
}

PyObject* interface(PyObject* self, PyObject *args) {
    PyArrayObject *input;
    if (!PyArg_ParseTuple(args, "O", &input)) {
        return NULL;
    }
    ndarray ndarr;
    ndarray_from_numpy(&ndarr, input);


    ndarray* ndarr2 = ndarray_multiply(&ndarr, 2.0);
    //ndarray_free(ndarr2);

    PyArrayObject* output = numpy_from_ndarray(ndarr2);
    PyArray_ENABLEFLAGS(output, NPY_ARRAY_OWNDATA);
    return (PyObject*)output;
}

static PyMethodDef methods[] =
{
     {"interface", interface, METH_VARARGS,
              "wtf"},
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


