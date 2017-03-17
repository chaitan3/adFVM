#ifndef INTERFACE_HPP
#define INTERFACE_HPP

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <string>
#include <map>

#include "common.hpp"

typedef map<string, map<string, string> > Boundary;

class Mesh {
    public:
        string caseDir;
        int nInternalFaces;
        int nFaces;
        int nBoundaryFaces;
        int nInternalCells;
        int nGhostCells;
        int nCells;

        iarr faces;
        arr points;
        iarr owner;
        iarr neighbour;

        arr normals;
        arr faceCentres;
        arr areas;
        iarr cellFaces;
        arr cellCentres;
        arr volumes;
        arr deltas;
        arr weights;

        //spmat sumOp;
        //spmat sumOpT;

        Boundary boundary;
        Boundary defaultBoundary;
        Boundary calculatedBoundary;

        PyObject* mesh; 
        PyObject* meshClass; 
        PyObject* meshModule; 

        //Mesh () {};
        Mesh (PyObject *);
        ~Mesh ();
        Mesh (string);
};

int getInteger(PyObject*, const string);
string getString(PyObject*, const string);

extern void putArray(PyObject *, const string, arr&);

template<typename dtype>
extern void getArray(PyObject *, const string, arrType<dtype> &);
//
//template<typename Derived>
//extern void getSpArray(PyObject *, const string, SparseMatrix<Derived> &);

Boundary getBoundary(PyObject*, const string);

#endif
