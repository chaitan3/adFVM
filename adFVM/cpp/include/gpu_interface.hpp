#ifndef GPU_INTERFACE_HPP
#define GPU_INTERFACE_HPP
#ifdef GPU
#include "interface.hpp"
#include "gpu.hpp"

template <typename dtype, integer shape1, integer shape2>
void getArray(PyArrayObject *array, gpuArrType<dtype, shape1, shape2> & tmp, int64_t id=-1) {
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
    gpuArrType<dtype, shape1, shape2> result(dims[0], data);
    if (id > -1) {
        result.id = id;
        result.shared();
    }
    tmp = move(result);
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
void Function_get_max_eigenvalue(std::vector<gpuArrType<float, 5, 5>*> phiP);

#endif
#endif
