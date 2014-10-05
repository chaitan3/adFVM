#include "interface.hpp"


Mesh::Mesh (string caseDir) {
    Py_Initialize();
    PyObject* mainmod = PyImport_AddModule("__main__");
    PyObject* maindict = PyModule_GetDict(mainmod);
    this->meshModule = PyImport_ImportModuleEx("mesh", maindict, maindict, NULL);
    assert(this->meshModule);
    this->meshClass = PyObject_GetAttrString(this->meshModule, "Mesh");
    assert(this->meshClass);
    PyObject *caseString = PyString_FromString(caseDir.c_str());
    PyObject *args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, caseString);
    this->mesh = PyObject_CallObject(this->meshClass, args);
    Py_DECREF(caseString);
    //Py_DECREF(args);
    assert(this->mesh);

    this->caseDir = getString(this->mesh, "case");
    this->nInternalFaces = getInteger(this->mesh, "nInternalFaces");
    this->nFaces = getInteger(this->mesh, "nFaces");
    this->nBoundaryFaces = getInteger(this->mesh, "nBoundaryFaces");
    this->nInternalCells = getInteger(this->mesh, "nInternalCells");
    this->nGhostCells = getInteger(this->mesh, "nGhostCells");
    this->nCells = getInteger(this->mesh, "nCells");

    getArray(this->mesh, "faces", this->faces);
    getArray(this->mesh, "points", this->points);
    getArray(this->mesh, "owner", this->owner);
    getArray(this->mesh, "neighbour", this->neighbour);

    getArray(this->mesh, "normals", this->normals);
    getArray(this->mesh, "faceCentres", this->faceCentres);
    getArray(this->mesh, "areas", this->areas);
    getArray(this->mesh, "cellFaces", this->cellFaces);
    getArray(this->mesh, "cellCentres", this->cellCentres);
    getArray(this->mesh, "volumes", this->volumes);

    getArray(this->mesh, "deltas", this->deltas);
    getArray(this->mesh, "weights", this->weights);

    getBoundary(this->mesh, "boundary", this->boundary);
    getBoundary(this->mesh, "calculatedBoundary", this->calculatedBoundary);
    getBoundary(this->mesh, "defaultBoundary", this->defaultBoundary);
}

Mesh::~Mesh () {
    Py_DECREF(this->mesh);
    Py_DECREF(this->meshClass);
    Py_DECREF(this->meshModule);
    if (Py_IsInitialized())
        Py_Finalize();
}


int getInteger(PyObject *mesh, const string attr) {
    PyObject *integer = PyObject_GetAttrString(mesh, attr.c_str());
    assert(integer);
    int result = (int)PyInt_AsLong(integer);
    Py_DECREF(integer);
    return result;
}

string getString(PyObject *mesh, const string attr) {
    PyObject *cstring = PyObject_GetAttrString(mesh, attr.c_str());
    assert(cstring);
    string result(PyString_AsString(cstring));
    Py_DECREF(cstring);
    return result;
}

template<typename Derived>
void getArray(PyObject *mesh, const string attr, MatrixBase<Derived> & tmp) {
    PyArrayObject *array = (PyArrayObject*) PyObject_GetAttrString(mesh, attr.c_str());
    assert(array);
    int nDims = PyArray_NDIM(array);
    npy_intp* dims = PyArray_DIMS(array);
    int rows = dims[1];
    int cols = dims[0];
    if (nDims == 1) {
        rows = 1;
    }
    typename Derived::Scalar *data = (typename Derived::Scalar *) PyArray_DATA(array);
    Map<Derived> result(data, rows, cols);
    tmp = result;
    Py_DECREF(array);
}

void getBoundary(PyObject *mesh, const string attr, Boundary& boundary) {
    PyObject *dict = PyObject_GetAttrString(mesh, attr.c_str());
    assert(dict);
    PyObject *key, *value;
    PyObject *key2, *value2;
    Py_ssize_t pos = 0;
    Py_ssize_t pos2 = 0;
    while (PyDict_Next(dict, &pos, &key, &value)) {
        string ckey = PyString_AsString(key);
        assert(value);
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
            else {
            }
            boundary[ckey][ckey2] = cvalue;
        }
    }
    Py_DECREF(dict);
}
