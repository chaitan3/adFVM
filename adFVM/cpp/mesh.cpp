#define NO_IMPORT_ARRAY
#include "mesh.hpp"
#include "parallel.hpp"

Mesh::Mesh (PyObject* meshObject) {
    this->mesh = meshObject;
    //Py_DECREF(args);
    assert(this->mesh);
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &this->nProcs);
    
    this->nCells = getInteger(this->mesh, "nCells");
    this->nFaces = getInteger(this->mesh, "nFaces");
    this->nInternalFaces = getInteger(this->mesh, "nInternalFaces");
    this->nInternalCells = getInteger(this->mesh, "nInternalCells");
    this->nBoundaryFaces = getInteger(this->mesh, "nBoundaryFaces");
    this->nGhostCells = getInteger(this->mesh, "nGhostCells");

    getMeshArray(this->mesh, "faces", this->faces);
    getMeshArray(this->mesh, "points", this->points);
    getMeshArray(this->mesh, "owner", this->owner);
    getMeshArray(this->mesh, "neighbour", this->neighbour);

    this->boundary = getMeshBoundary(this->mesh, "boundary");
    for (auto& patch: this->boundary) {
        string patchID = patch.first;
        auto& patchInfo = patch.second;
        integer startFace = stoi(patchInfo.at("startFace"));
        integer nFaces = stoi(patchInfo.at("nFaces"));
        this->boundaryFaces[patchID] = make_pair(startFace, nFaces);
    }
}

void Mesh::init () {
    this->nLocalCells = getInteger(this->mesh, "nLocalCells");
    this->nLocalFaces = this->nFaces - (this->nCells-this->nLocalCells);
    this->nLocalPatches = getInteger(this->mesh, "nLocalPatches");
    this->nRemotePatches = getInteger(this->mesh, "nRemotePatches");
    this->tags = getTags(this->mesh, "tags");

    getMeshArray(this->mesh, "cellNeighboursMatOp", this->cellNeighbours);
    getMeshArray(this->mesh, "areas", this->areas);
    getMeshArray(this->mesh, "deltas", this->deltas);
    getMeshArray(this->mesh, "cellFaces", this->cellFaces);
    getMeshArray(this->mesh, "volumes", this->volumes);
    if (this->rank == 0) {
        std::cout << "Initializing C++ interface" << endl;
    }
}



Mesh::~Mesh () {
    //Py_DECREF(this->mesh);
    //Py_DECREF(this->meshClass);
    //Py_DECREF(this->meshModule);
}

int getInteger(PyObject *mesh, const string attr) {
    PyObject *in = PyObject_GetAttrString(mesh, attr.c_str());
    assert (in != NULL);
    int result = (int)PyInt_AsLong(in);
    //cout << attr << " " << result << endl;
    Py_DECREF(in);
    return result;
}

map<string, integer> getTags(PyObject *mesh, const string attr) {
    PyObject *dict = PyObject_GetAttrString(mesh, attr.c_str());
    assert (dict != NULL);
    map<string, integer> tags;
    PyObject *key2, *value2;
    Py_ssize_t pos2 = 0;
    while (PyDict_Next(dict, &pos2, &key2, &value2)) {
        string ckey2 = PyString_AsString(key2);
        tags[ckey2] = PyInt_AsLong(value2);
    }
    return tags;
}

Boundary getMeshBoundary(PyObject *mesh, const string attr) {
    PyObject *dict = PyObject_GetAttrString(mesh, attr.c_str());
    assert (dict != 0);
    //cout << attr << " " << dict << endl;
    return getBoundary(dict);
}

//void getDict(PyObject* dict, map<string, string>& cDict) {
//    PyObject *key2, *value2;
//    Py_ssize_t pos2 = 0;
//    while (PyDict_Next(dict, &pos2, &key2, &value2)) {
//        string ckey2 = PyString_AsString(key2);
//        string cvalue;
//        if (PyInt_Check(value2)) {
//            int ivalue = (int)PyInt_AsLong(value2);
//            cvalue = to_string(ivalue);
//        }
//        if (PyFloat_Check(value2)) {
//            scalar ivalue = PyFloat_AsDouble(value2);
//            cvalue = to_string(ivalue);
//        }
//        else if (PyString_Check(value2)) {
//            cvalue = PyString_AsString(value2);
//        }
//        else if (PyArray_Check(value2)) {
//            Py_INCREF(value2);
//            PyArrayObject* val = (PyArrayObject*) value2;
//            char* data = (char *) PyArray_DATA(val);
//            int size = PyArray_NBYTES(val);
//            cvalue = string(data, size);
//        }
//        cDict[ckey2] = cvalue;
//    }
//}

Boundary getBoundary(PyObject *dict) {
    assert(dict);
    PyObject *key, *value;
    PyObject *key2, *value2;
    Py_ssize_t pos = 0;
    Py_ssize_t pos2 = 0;
    Boundary boundary;
    while (PyDict_Next(dict, &pos, &key, &value)) {
        string ckey = PyString_AsString(key);
        assert(value);
        pos2 = 0;
        while (PyDict_Next(value, &pos2, &key2, &value2)) {
            string ckey2 = PyString_AsString(key2);
            string cvalue;
            if (PyInt_Check(value2)) {
                int ivalue = (int)PyInt_AsLong(value2);
                cvalue = to_string(ivalue);
            }
            else if (PyString_Check(value2)) {
                cvalue = PyString_AsString(value2);
            }
            else if (ckey2[0] == '_' || ckey2 == "loc_neighbourIndices") {
                Py_INCREF(value2);
                PyArrayObject* val = (PyArrayObject*) value2;
                char* data = (char *) PyArray_DATA(val);
                int size = PyArray_NBYTES(val);
                cvalue = string(data, size);
            }
            boundary[ckey][ckey2] = cvalue;
        }
    }
    return boundary;
}


