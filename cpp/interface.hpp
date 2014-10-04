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

        imat faces;
        mat points;
        imat owner;
        imat neighbour;

        mat normals;
        mat faceCentres;
        mat areas;
        imat cellFaces;
        mat cellCentres;
        mat volumes;
        mat deltas;
        mat weights;

        Boundary boundary;
        Boundary defaultBoundary;
        Boundary calculatedBoundary;

        PyObject* mesh; 
        PyObject* meshClass; 
        PyObject* meshModule; 

        Mesh () {};
        ~Mesh ();
        Mesh (string);
};

int getInteger(PyObject*, const string);
string getString(PyObject*, const string);
template<typename Derived>
extern void getArray(PyObject *, const string, MatrixBase<Derived> &);
void getBoundary(PyObject*, const string, Boundary&);

#endif
