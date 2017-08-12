#ifndef INTERFACE_HPP
#define INTERFACE_HPP

#define PY_ARRAY_UNIQUE_SYMBOL cool_ARRAY_API
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <string>
#include <map>
#include "mpi.h"

#if PY_MAJOR_VERSION >= 3
#define PY3 1
#define PyInt_AsLong PyLong_AsLong
#define PyInt_Check PyLong_Check
char* PyString_AsString(PyObject* result);
#define PyString_Check PyUnicode_Check
#endif

#include "common.hpp"

scalar getMaxEigenvalue(arrType<scalar, 5, 5>& phi, vec& eigPhi);

template <typename dtype, integer shape1, integer shape2>
void getArray(PyArrayObject *array, arrType<dtype, shape1, shape2> & tmp) {
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
    dtype* data = tmp.data;
    tmp.ownData = false;
    PyObject *array;
    if (typeid(dtype) == typeid(double)) {
        array = PyArray_SimpleNewFromData(2, shape, NPY_DOUBLE, data);
    } else if (typeid(dtype) == typeid(float)) {
        array = PyArray_SimpleNewFromData(2, shape, NPY_FLOAT, data);
    } else {
        array = PyArray_SimpleNewFromData(2, shape, NPY_INT32, data);
    }
    PyArray_ENABLEFLAGS((PyArrayObject*)array, NPY_ARRAY_OWNDATA);
    tmp.dec_mem();
    return array;
}
template <typename dtype, integer shape1, integer shape2>
PyObject* putArray(arrType<dtype, shape1, shape2> &tmp) {
    npy_intp shape[3] = {tmp.shape, shape1, shape2};
    dtype* data = tmp.data;
    tmp.ownData = false;
    PyObject *array;
    if (typeid(dtype) == typeid(double)) {
        array = PyArray_SimpleNewFromData(3, shape, NPY_DOUBLE, data);
    } else if (typeid(dtype) == typeid(float)) {
        array = PyArray_SimpleNewFromData(3, shape, NPY_FLOAT, data);
    } else {
        array = PyArray_SimpleNewFromData(3, shape, NPY_INT32, data);
    }
    PyArray_ENABLEFLAGS((PyArrayObject*)array, NPY_ARRAY_OWNDATA);
    tmp.dec_mem();
    return array;
}

#ifdef GPU
template <typename dtype, integer shape1, integer shape2>
void getArray(PyArrayObject *array, gpuArrType<dtype, shape1, shape2> & tmp, const string& attr) {
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

    //arrType<dtype, shape1, shape2> test(dims[0], data);
    //test.info();

    dtype *store = NULL;
    if (tmp.staticVariable && mem.refs.count(attr)) {
        store = (dtype *)mem.refs.at(attr);
        gpuArrType<dtype, shape1, shape2> result(dims[0], store);
        tmp = move(result);
    } else {
        gpuArrType<dtype, shape1, shape2> result(dims[0], store);
        result.toDevice(data);
        if (tmp.staticVariable) {
            result.ownData = false;
            mem.refs[attr] = (void *)result.data;
        }
        tmp = move(result);
    }
    //result.info();
}

template <typename dtype, integer shape1>
PyObject* putArray(gpuArrType<dtype, shape1> &tmp) {
    npy_intp shape[2] = {tmp.shape, shape1};
    dtype* data = tmp.toHost();
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
PyObject* putArray(gpuArrType<dtype, shape1, shape2> &tmp) {
    npy_intp shape[3] = {tmp.shape, shape1, shape2};
    dtype* data = tmp.toHost();

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
#endif

#endif
