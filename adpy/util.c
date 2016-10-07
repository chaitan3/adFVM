#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>


typedef double scalar_t;
typedef int64_t index_t;
typedef int32_t int_t;


//#define FOR_LOOP (i, n) for(index_t i=0;i<n; i++)

typedef struct {
    char * data;
    index_t *shape;
    index_t *strides;

    int dims;
    int type;
    int bytes;
    index_t size;
} ndarray;

ndarray* ndarray_from_numpy(PyArrayObject *arr) {
    ndarray *ndarr = malloc(sizeof(ndarray));
    ndarr -> data = arr -> data;
    ndarr -> dims = arr -> nd;
    ndarr -> shape = arr -> dimensions;
    ndarr -> strides = arr -> strides;
    ndarr -> type = arr -> descr -> type_num;
    ndarr -> bytes = arr -> descr -> elsize;
    ndarr -> size = 1;
    for (index_t i = 0; i < ndarr -> dims; i++) {
        ndarr -> size *= ndarr -> shape[i];
    }
    //printf("%d %f\n", ndarr->size, ((double*)ndarr->data)[0]);
    return ndarr;
}

ndarray* ndarray_alloc_new(index_t *shape, int dims, int type) {
    ndarray *ndarr = malloc(sizeof(ndarray));
    ndarr -> shape = shape;
    ndarr -> dims = dims;
    ndarr -> type = type;

    ndarr -> bytes = 8;

    ndarr -> size = 1;
    if (dims > 0) {
        ndarr -> strides = malloc(sizeof(index_t)*dims);
        ndarr -> strides [dims-1] = ndarr -> bytes;
        for (index_t i = 0; i < dims; i++) {
            index_t j = dims-i-2;
            if (j >= 0) {
                ndarr -> strides[j] = ndarr -> strides[j+1] * ndarr -> shape[j+1];
            }
            ndarr -> size *= ndarr -> shape[i];
        }
    } else {
        ndarr -> strides = NULL;
    }
    ndarr -> data = malloc(ndarr->size*ndarr->bytes);
    return ndarr; 
}

ndarray* ndarray_alloc_from(ndarray* ndarr_ref) {
    ndarray *ndarr = malloc(sizeof(ndarray));
    memcpy(ndarr, ndarr_ref, sizeof(ndarray));

    index_t shape_size = sizeof(index_t)*ndarr->dims;
    ndarr -> shape = malloc(shape_size);
    ndarr -> strides = malloc(shape_size);
    memcpy(ndarr->shape, ndarr_ref->shape, shape_size);
    memcpy(ndarr->strides, ndarr_ref->strides, shape_size);

    index_t data_size = ndarr -> size * ndarr -> bytes;
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
    free(ndarr->strides);
    free(ndarr);
}

ndarray* ndarray_multiply(ndarray *ndarr, float constant) {
    ndarray* ndarr2 = ndarray_alloc_from(ndarr);
    float *data0 = ndarr -> data;
    float *data1 = ndarr2 -> data;
    int64_t i;
    for (i = 0; i < ndarr -> size; i++) {
        data1[i] = data0[i]*constant;
    }
    //printf("data1[0] %f\n", ((float *)ndarr2->data)[0]);
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


