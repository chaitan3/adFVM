#ifndef INTERFACE_HPP
#define INTERFACE_HPP

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
        void init();
        ~Mesh ();
        Mesh (string);
};

int getInteger(PyObject*, const string);
string getString(PyObject*, const string);
void getDict(PyObject* dict, map<string, string>& cDict);
map<string, integer> getTags(PyObject *mesh, const string attr);

//template<typename dtype, integer shape1, integer shape2>
// PyObject * putArray(arrType<dtype, shape1, shape2>&);
//
//template<typename dtype, integer shape1, integer shape2>
// void getMeshArray(PyObject *, const string, arrType<dtype, shape1, shape2> &);
//
//template<typename dtype, integer shape1>
// void getArray(PyArrayObject *, arrType<dtype, shape1> &);
//
//template<typename Derived>
//extern void getSpArray(PyObject *, const string, SparseMatrix<Derived> &);

Boundary getBoundary(PyObject*);
Boundary getMeshBoundary(PyObject *mesh, const string attr);

#endif
