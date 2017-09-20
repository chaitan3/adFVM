#ifndef INTERFACE_HPP
#define INTERFACE_HPP

#define PY_ARRAY_UNIQUE_SYMBOL cool_ARRAY_API
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "mpi.h"

#if PY_MAJOR_VERSION >= 3
#define PY3 1
#define PyInt_AsLong PyLong_AsLong
#define PyInt_Check PyLong_Check
#define PyString_Check PyUnicode_Check
char* PyString_AsString(PyObject* result);

#endif

#include "common.hpp"
#include "gpu.hpp"

#define _GET_MODULE(name, mod) name##mod
#define GET_MODULE(name, mod) _GET_MODULE(name, mod)


PyObject* PyTuple_CreateNone(int);
map<string, int> PyOptions_Parse(PyObject*);

scalar getMaxEigenvalue(arrType<scalar, 5, 5>& phi, vec& eigPhi);

template <template<typename, integer, integer, integer> class derivedArrType, typename dtype, integer shape1, integer shape2=1, integer shape3=1>
void getArray(PyArrayObject *array, derivedArrType<dtype, shape1, shape2, shape3>& tmp, bool keepMemory=false, int64_t id=0) {
    static_assert(shape3 == 1, "shape3 exceeded");
    assert(array);
    int nDims = PyArray_NDIM(array);
    npy_intp* dims = PyArray_DIMS(array);
    if (nDims > 1) {
        assert(dims[1] == shape1);
        //cout << dims[1] << " " << shape1 << endl;
    }
    if (nDims > 2) {
        assert(dims[2] == shape2);
        //cout << dims[2] << " " << shape2 << endl;
    }
    assert(PyArray_IS_C_CONTIGUOUS(array));
    assert(PyArray_ITEMSIZE(array) == sizeof(dtype));

    dtype *data = (dtype *) PyArray_DATA(array);
    derivedArrType<dtype, shape1, shape2, shape3> result(dims[0], data);
    result.keepMemory = keepMemory;
    result.id = id;
    result.toDevice(result.data);
    
    //cout << rows << " " << cols << endl;
    //if ((typeid(dtype) != type(uscalar)) && (typeid(dtype) != typeid(integer))) {
    tmp = move(result);
}

template <template<typename, integer, integer, integer> class derivedArrType, typename dtype, integer shape1, integer shape2=1, integer shape3=1>
PyObject* putArray(derivedArrType<dtype, shape1, shape2, shape3> &tmp, bool reuseMemory=true) {
    static_assert(shape3 == 1, "shape3 exceeded");
    npy_intp shape[3] = {tmp.shape, shape1, shape2};
    dtype* data;
    int size = 3;
    if (shape2 == 1) {
        size = 2;
    }
    if (reuseMemory) {
        // assert arrType
        data = tmp.data;
        tmp.ownData = false;
    } else {
        data = tmp.toHost();
    }
    PyObject *array;
    if (typeid(dtype) == typeid(double)) {
        array = PyArray_SimpleNewFromData(size, shape, NPY_DOUBLE, data);
    } else if (typeid(dtype) == typeid(float)) {
        array = PyArray_SimpleNewFromData(size, shape, NPY_FLOAT, data);
    } else {
        array = PyArray_SimpleNewFromData(size, shape, NPY_INT32, data);
    }
    PyArray_ENABLEFLAGS((PyArrayObject*)array, NPY_ARRAY_OWNDATA);
    return array;
}
#ifndef GPU
    void Function_get_max_eigenvalue(std::vector<arrType<scalar, 5, 5>*> phiP);
#else
    void Function_get_max_eigenvalue(std::vector<gpuArrType<scalar, 5, 5>*> phiP);
#endif

#endif
