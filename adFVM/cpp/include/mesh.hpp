#ifndef MESH_HPP
#define MESH_HPP

#include "interface.hpp"

typedef map<string, map<string, string> > Boundary;
class Mesh {
    public:
        int nCells;
        int nFaces;
        int nInternalCells;
        int nInternalFaces;
        int nBoundaryFaces;
        int nGhostCells;
        int nLocalPatches;
        int nRemotePatches;
        int nLocalCells;
        int nLocalFaces;
        int nProcs, rank, localRank;

        arrType<integer, 5> faces;
        mat points;
        ivec owner;
        ivec neighbour;

        mat normals;
        vec areas, weights, deltas, volumes;
        mat deltasUnit;
        arrType<scalar, 2> linearWeights;
        arrType<scalar, 2, 3> quadraticWeights;
        mat faceCentres;
        mat cellCentres;
        arrType<integer, 6> cellFaces;
        arrType<integer, 8> cells;
        arrType<integer, 6> cellNeighbours;

        Boundary boundary;
        map<string, pair<integer, integer>> boundaryFaces;
        map<string, integer> tags;

        PyObject* mesh; 

        Mesh (PyObject *);
        void init();
        void build();
        void buildBeforeWrite();
        ~Mesh ();
};

int getInteger(PyObject*, const string);
map<string, integer> getTags(PyObject *mesh, const string attr);
template <typename dtype, integer shape1, integer shape2>
void getMeshArray(PyObject *mesh, const string attr, arrType<dtype, shape1, shape2>& tmp) {
    PyArrayObject *array = (PyArrayObject*) PyObject_GetAttrString(mesh, attr.c_str());
    //cout << attr << " " << array << endl;
    //cout << attr << " " << PyArray_DESCR(array)->elsize << endl;
    assert (array != NULL);
    getArray(array, tmp);
    Py_DECREF(array);
}

Boundary getBoundary(PyObject*);
Boundary getMeshBoundary(PyObject *mesh, const string attr);

PyObject* finalSolver(PyObject *self, PyObject *args);
PyObject* initSolver(PyObject *self, PyObject *args);
PyObject* viscosity(PyObject *self, PyObject *args);

extern Mesh* meshp;

#endif
