#ifndef MESH_HPP
#define MESH_HPP

#include "interface.hpp"

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
        int nProcs, rank;

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

        PyObject* mesh; 

        Mesh (PyObject *);
        void init();
        void build();
        ~Mesh ();
};

int getInteger(PyObject*, const string);
map<string, integer> getTags(PyObject *mesh, const string attr);
template<typename dtype, integer shape1, integer shape2>
 void getMeshArray(PyObject *, const string, arrType<dtype, shape1, shape2> &);

Boundary getBoundary(PyObject*);
Boundary getMeshBoundary(PyObject *mesh, const string attr);

PyObject* finalSolver(PyObject *self, PyObject *args);
PyObject* initSolver(PyObject *self, PyObject *args);
PyObject* viscosity(PyObject *self, PyObject *args);

extern Mesh* meshp;

#endif
