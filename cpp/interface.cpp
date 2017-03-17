#include "interface.hpp"
#include "density.cpp"

Mesh *mesh;
RCF rcf(mesh);

static PyObject* initSolver(PyObject *self, PyObject *args) {

    PyObject *meshObject = PyTuple_GetItem(args, 0);
    mesh = new Mesh(meshObject);
    return Py_None;
}

static PyObject* forwardSolver(PyObject *self, PyObject *args) {

    PyObject *rho, *rhoU, *rhoE;
    PyArg_ParseTuple(args, "OOO", &rho, &rhoU, &rhoE);
    return Py_BuildValue("(OOO)", rho, rhoU, rhoE);
}

PyMODINIT_FUNC
initadFVMcpp(void)
{
    PyObject *m;

    static PyMethodDef Methods[] = {
        {"forward",  forwardSolver, METH_VARARGS, "boo"},
        {"init",  initSolver, METH_VARARGS,
             "Execute a shell command."},
        {NULL, NULL, 0, NULL}        /* Sentinel */
    };

    m = Py_InitModule("adFVMcpp", Methods);
    if (m == NULL)
        return;

    //SpamError = PyErr_NewException("spam.error", NULL, NULL);
    //Py_INCREF(SpamError);
    //PyModule_AddObject(m, "error", SpamError);
}



Mesh::Mesh (PyObject* meshObject) {
    this->mesh = meshObject;
    //Py_DECREF(args);
    assert(this->mesh);
    std::cout << "Initializing C++ interface" << endl;

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

template <typename dtype>
void getArray(PyObject *mesh, const string attr, arrType<dtype> & tmp) {
    PyArrayObject *array = (PyArrayObject*) PyObject_GetAttrString(mesh, attr.c_str());
    assert(array);
    int nDims = PyArray_NDIM(array);
    npy_intp* dims = PyArray_DIMS(array);
    int rows = dims[1];
    int cols = dims[0];
    //cout << attr << " " << rows << " " << cols << endl;
    if (nDims == 1) {
        rows = 1;
    }
    dtype *data = (dtype *) PyArray_DATA(array);
    //cout << rows << " " << cols << endl;
    integer shape[NDIMS] = {cols, rows, 1, 1};
    arrType<dtype>* result = new arrType<dtype>(shape, data);
    tmp = *result;
    Py_DECREF(array);
}

void putArray(PyObject *mesh, const string attr, arr &tmp) {
    PyArrayObject *array = (PyArrayObject*) PyObject_GetAttrString(mesh, attr.c_str());
    assert(array);
    double *data = (double *) PyArray_DATA(array);
    //memcpy(data, tmp.data(), tmp.size() * sizeof(double));
    Py_DECREF(array);
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
