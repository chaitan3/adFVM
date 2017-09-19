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

#define _GET_MODULE(name, mod) name##mod
#define GET_MODULE(name, mod) _GET_MODULE(name, mod)

scalar getMaxEigenvalue(arrType<scalar, 5, 5>& phi, vec& eigPhi);

template <typename dtype, integer shape1, integer shape2>
void getArray(PyArrayObject *array, arrType<dtype, shape1, shape2> & tmp, const string& attr="") {
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

    arrType<dtype, shape1, shape2> result(dims[0], (dtype *) PyArray_DATA(array));
    //cout << rows << " " << cols << endl;
    //if ((typeid(dtype) != type(uscalar)) && (typeid(dtype) != typeid(integer))) {
    tmp = move(result);
}

template <typename dtype, integer shape1>
PyObject* putArray(arrType<dtype, shape1> &tmp) {
    npy_intp shape[2] = {tmp.shape, shape1};
    dtype* data = new dtype[tmp.size];
    memcpy(data, tmp.data, tmp.bufSize);
    PyObject *array;
    if (typeid(dtype) == typeid(double)) {
        array = PyArray_SimpleNewFromData(2, shape, NPY_DOUBLE, data);
    } else if (typeid(dtype) == typeid(float)) {
        array = PyArray_SimpleNewFromData(2, shape, NPY_FLOAT, data);
    } else {
        array = PyArray_SimpleNewFromData(2, shape, NPY_INT32, data);
    }
    PyArray_ENABLEFLAGS((PyArrayObject*)array, NPY_ARRAY_OWNDATA);
    return array;
}
template <typename dtype, integer shape1, integer shape2>
PyObject* putArray(arrType<dtype, shape1, shape2> &tmp) {
    npy_intp shape[3] = {tmp.shape, shape1, shape2};
    dtype* data = new dtype[tmp.size];
    memcpy(data, tmp.data, tmp.bufSize);
    PyObject *array;
    if (typeid(dtype) == typeid(double)) {
        array = PyArray_SimpleNewFromData(3, shape, NPY_DOUBLE, data);
    } else if (typeid(dtype) == typeid(float)) {
        array = PyArray_SimpleNewFromData(3, shape, NPY_FLOAT, data);
    } else {
        array = PyArray_SimpleNewFromData(3, shape, NPY_INT32, data);
    }
    PyArray_ENABLEFLAGS((PyArrayObject*)array, NPY_ARRAY_OWNDATA);
    return array;
}

#ifndef GPU
    void Function_get_max_eigenvalue(std::vector<arrType<scalar, 5, 5>*> phiP);
#endif

#endif
