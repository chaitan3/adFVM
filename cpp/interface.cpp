#include "interface.hpp"


Mesh::Mesh (string caseDir) {
    Py_Initialize();
    PyObject* mainmod = PyImport_AddModule("__main__");
    PyObject* maindict = PyModule_GetDict(mainmod);
    this->meshModule = PyImport_ImportModuleEx("mesh", maindict, maindict, NULL);
    assert(this->meshModule);
    this->meshClass = PyObject_GetAttrString(this->meshModule, "Mesh");
    assert(this->meshClass);
    PyObject *meshCreate = PyObject_GetAttrString(this->meshClass, "create");
    PyObject *caseString = PyString_FromString(caseDir.c_str());
    PyObject *args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, caseString);
    this->mesh = PyObject_CallObject(meshCreate, args);
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

    getSpArray(this->mesh, "sumOp", this->sumOp);
    this->sumOpT = this->sumOp.transpose();

    this->boundary = getBoundary(this->mesh, "boundary");
    this->calculatedBoundary = getBoundary(this->mesh, "calculatedBoundary");
    this->defaultBoundary = getBoundary(this->mesh, "defaultBoundary");
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

template <typename Derived>
void getArray(PyObject *mesh, const string attr, DenseBase<Derived> & tmp) {
    PyArrayObject *array = (PyArrayObject*) PyObject_GetAttrString(mesh, attr.c_str());
    assert(array);
    int nDims = PyArray_NDIM(array);
    npy_intp* dims = PyArray_DIMS(array);
    int rows = dims[1];
    int cols = dims[0];
    cout << attr << " " << rows << " " << cols << endl;
    if (nDims == 1) {
        rows = 1;
    }
    typename Derived::Scalar *data = (typename Derived::Scalar *) PyArray_DATA(array);
    //cout << rows << " " << cols << endl;
    Map<Derived> result(data, rows, cols);
    tmp = result;
    Py_DECREF(array);
}

void putArray(PyObject *mesh, const string attr, arr &tmp) {
    PyArrayObject *array = (PyArrayObject*) PyObject_GetAttrString(mesh, attr.c_str());
    assert(array);
    double *data = (double *) PyArray_DATA(array);
    memcpy(data, tmp.data(), tmp.size() * sizeof(double));
    Py_DECREF(array);
}



template <typename Derived>
void getSpArray(PyObject *mesh, const string attr, SparseMatrix<Derived> & tmp) {
    PyObject *sparse = PyObject_GetAttrString(mesh, attr.c_str());
    assert(sparse);
    PyObject *shape = PyObject_GetAttrString(sparse, "shape");
    PyArrayObject *data = (PyArrayObject*) PyObject_GetAttrString(sparse, "data");
    PyArrayObject *indices = (PyArrayObject*) PyObject_GetAttrString(sparse, "indices");
    PyArrayObject *indptr = (PyArrayObject*) PyObject_GetAttrString(sparse, "indptr");
    Derived *cdata = (Derived*) PyArray_DATA(data);
    int32_t *cindices = (int32_t *)PyArray_DATA(indices);
    int32_t *cindptr = (int32_t *)PyArray_DATA(indptr);

    PyObject *pyRows = PyTuple_GetItem(shape, 0);
    PyObject *pyCols = PyTuple_GetItem(shape, 1);
    int rows = PyInt_AsLong(pyRows);
    int cols = PyInt_AsLong(pyCols);
    //cout << attr << " " << rows << " " << cols << endl;

    tmp = spmat(rows, cols);
    tmp.reserve(2*cols);
    // fill
    for (int i = 0; i < rows; i++) {
        for (int j = cindptr[i]; j < cindptr[i+1] ; j++) {
            tmp.insert(i, cindices[j]) = cdata[j];
        }
    }
    //
    //tmp.makeCompressed();
    //cout << rows << " " << cols << endl;
    Py_DECREF(shape);
    Py_DECREF(pyRows);
    Py_DECREF(pyCols);
    Py_DECREF(data);
    Py_DECREF(indices);
    Py_DECREF(indptr);
    Py_DECREF(sparse);
}



Boundary getBoundary(PyObject *mesh, const string attr) {
    PyObject *dict = PyObject_GetAttrString(mesh, attr.c_str());
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
            else {
            }
            boundary[ckey][ckey2] = cvalue;
        }
    }
    Py_DECREF(dict);
    return boundary;
}
