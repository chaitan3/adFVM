#ifndef INTERFACE_HPP
#define INTERFACE_HPP

#define PY_ARRAY_UNIQUE_SYMBOL cool_ARRAY_API
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <string>
#include <map>
#include "mpi.h"

#include "common.hpp"

typedef map<string, map<string, string> > Boundary;

class Mesh {
    public:
        string caseDir;
        int nCells;
        int nFaces;
        int nInternalCells;
        int nInternalFaces;
        int nBoundaryFaces;
        int nGhostCells;
        int nLocalFaces;
        int nLocalCells;
        int nLocalPatches;
        int nRemotePatches;
        int nProcs;

        imat faces;
        mat points;
        ivec owner;
        ivec neighbour;

        arrType<integer, 6> cellFaces;
        arrType<integer, 6> cellNeighbours;
        vec volumes;

        vec areas, weights, deltas;
        vec volumesL, volumesR;
        mat normals;
        arrType<scalar, 2> linearWeights;
        arrType<scalar, 2, 3> quadraticWeights;

        mat deltasUnit;
        mat faceCentres;
        mat cellCentres;

        Boundary boundary;
        map<string, pair<integer, integer>> boundaryFaces;
        map<string, integer> tags;
        Boundary defaultBoundary;
        Boundary calculatedBoundary;

        PyObject* mesh; 
        PyObject* meshClass; 
        PyObject* meshModule; 

        //Mesh () {};
        Mesh (PyObject *);
        Mesh (const Mesh& mesh);
        void init();
        void reset();
        ~Mesh ();
        Mesh (string);
};

int getInteger(PyObject*, const string);
string getString(PyObject*, const string);
void getDict(PyObject* dict, map<string, string>& cDict);
map<string, integer> getTags(PyObject *mesh, const string attr);
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

    arrType<dtype, shape1, shape2> result(dims[0], (dtype *) PyArray_DATA(array));
    //cout << rows << " " << cols << endl;
    //if ((typeid(dtype) != type(uscalar)) && (typeid(dtype) != typeid(integer))) {
    tmp = move(result);
    //result.ownData = false;
}

template <typename dtype, integer shape1>
PyObject* putArray(arrType<dtype, shape1> &tmp) {
    npy_intp shape[2] = {tmp.shape, shape1};
    scalar* data = tmp.data;
    tmp.ownData = false;
    PyObject* array = PyArray_SimpleNewFromData(2, shape, NPY_DOUBLE, data);
    PyArray_ENABLEFLAGS((PyArrayObject*)array, NPY_ARRAY_OWNDATA);
    return array;
}
template <typename dtype, integer shape1, integer shape2>
PyObject* putArray(arrType<dtype, shape1, shape2> &tmp) {
    npy_intp shape[3] = {tmp.shape, shape1, shape2};
    scalar* data = tmp.data;
    tmp.ownData = false;
    PyObject* array = PyArray_SimpleNewFromData(3, shape, NPY_DOUBLE, data);
    PyArray_ENABLEFLAGS((PyArrayObject*)array, NPY_ARRAY_OWNDATA);
    return array;
}


//template<typename dtype, integer shape1, integer shape2>
// PyObject * putArray(arrType<dtype, shape1, shape2>&);
//
//template <typename dtype, integer shape1>
// PyObject* putArray(arrType<dtype, shape1> &tmp);
//
//template<typename dtype, integer shape1, integer shape2>
// void getArray(PyArrayObject *, arrType<dtype, shape1, shape2> &);

template<typename dtype, integer shape1, integer shape2>
 void getMeshArray(PyObject *, const string, arrType<dtype, shape1, shape2> &);

//
//template<typename Derived>
//extern void getSpArray(PyObject *, const string, SparseMatrix<Derived> &);
//

PyObject* finalSolver(PyObject *self, PyObject *args);
PyObject* initSolver(PyObject *self, PyObject *args);
PyObject* viscosity(PyObject *self, PyObject *args);

void Function_mpi_init();
template <typename dtype, integer shape1, integer shape2>
void Function_mpi(std::vector<arrType<dtype, shape1, shape2>*> phiP);
void Function_mpi_end();
void Function_mpi_allreduce(std::vector<vec*> vals);

Boundary getBoundary(PyObject*);
Boundary getMeshBoundary(PyObject *mesh, const string attr);
extern Mesh* meshp;

#endif
