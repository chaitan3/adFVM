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

        iarr faces;
        uarr points;
        iarr owner;
        iarr neighbour;

        uarr normals;
        uarr faceCentres;
        uarr areas;
        iarr cellFaces;
        uarr cellCentres;
        uarr volumes;
        uarr deltas;
        uarr deltasUnit;
        uarr weights;
        uarr linearWeights;
        uarr quadraticWeights;

        //spmat sumOp;
        //spmat sumOpT;

        Boundary boundary;
        map<string, pair<integer, integer>> boundaryFaces;
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

template<typename dtype>
PyObject * putArray(arrType<dtype>&);

template<typename dtype>
extern void getMeshArray(PyObject *, const string, arrType<dtype> &);

template<typename dtype>
extern void getArray(PyArrayObject *, arrType<dtype> &);
//
//template<typename Derived>
//extern void getSpArray(PyObject *, const string, SparseMatrix<Derived> &);

Boundary getBoundary(PyObject*);
Boundary getMeshBoundary(PyObject *mesh, const string attr);

#endif
