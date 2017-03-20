#include "interface.hpp"
#include "timestep.hpp"
#include "density.hpp"

RCF* rcf;
#ifdef ADIFF
    auto& tape = codi::RealReverse::getGlobalTape();    
#endif

static PyObject* initSolver(PyObject *self, PyObject *args) {

    PyObject *meshObject = PyTuple_GetItem(args, 0);

    Mesh *mesh = new Mesh(meshObject);
    rcf = new RCF();
    rcf->setMesh(mesh);
    for (integer i = 1; i < 4; i++) {
        PyObject *boundaryObject = PyTuple_GetItem(args, i);
        rcf->boundaries[i-1] = getBoundary(boundaryObject);
    }
    return Py_None;
}

static PyObject* forwardSolver(PyObject *self, PyObject *args) {

    //cout << "forward 1" << endl;
    PyObject *rhoObject, *rhoUObject, *rhoEObject;
    uscalar t, dt;
    PyArg_ParseTuple(args, "OOOdd", &rhoObject, &rhoUObject, &rhoEObject, &dt, &t);

    arr rho, rhoU, rhoE;
    getArray((PyArrayObject *)rhoObject, rho);
    getArray((PyArrayObject *)rhoUObject, rhoU);
    getArray((PyArrayObject *)rhoEObject, rhoE);
    //cout << "forward 2" << endl;

    //cout << "forward 3" << endl;
    #ifdef ADIFF
        tape.reset();
        tape.setActive();
        rho.adInit(tape);
        rhoU.adInit(tape);
        rhoE.adInit(tape);
    #endif

    arr rhoN(rho.shape);
    arr rhoUN(rhoU.shape);
    arr rhoEN(rhoE.shape);
    rhoN.ownData = false;
    rhoUN.ownData = false;
    rhoEN.ownData = false;
    timeStepper(rcf, rho, rhoU, rhoE, rhoN, rhoUN, rhoEN, t, dt);
    //cout << "forward 4" << endl;
    #ifdef ADIFF
        scalar obj = 0.;
        for (int i = 0; i < rho.size; i++) {
            obj += rho(i);
        }
        tape.registerOutput(obj);
        tape.setPassive();

        obj.setGradient(1.0);
        tape.evaluate();
        cout << "evaluated tape" << endl;
    #endif


    PyObject *rhoNObject, *rhoUNObject, *rhoENObject;
    rhoNObject = putArray(rhoN);
    rhoUNObject = putArray(rhoUN);
    rhoENObject = putArray(rhoEN);
    //cout << "forward 5" << endl;
    
    return Py_BuildValue("(OOO)", rhoNObject, rhoUNObject, rhoENObject);
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
    import_array();

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
    this->nLocalCells = getInteger(this->mesh, "nLocalCells");
    this->nLocalFaces = this->nFaces - (this->nCells-this->nLocalCells);

    getMeshArray(this->mesh, "faces", this->faces);
    getMeshArray(this->mesh, "points", this->points);
    getMeshArray(this->mesh, "owner", this->owner);
    getMeshArray(this->mesh, "neighbour", this->neighbour);

    getMeshArray(this->mesh, "normals", this->normals);
    getMeshArray(this->mesh, "faceCentres", this->faceCentres);
    getMeshArray(this->mesh, "areas", this->areas);
    getMeshArray(this->mesh, "cellFaces", this->cellFaces);
    getMeshArray(this->mesh, "cellCentres", this->cellCentres);
    getMeshArray(this->mesh, "volumes", this->volumes);

    getMeshArray(this->mesh, "deltas", this->deltas);
    getMeshArray(this->mesh, "weights", this->weights);
    getMeshArray(this->mesh, "linearWeights", this->linearWeights);
    getMeshArray(this->mesh, "quadraticWeights", this->quadraticWeights);

    this->boundary = getMeshBoundary(this->mesh, "boundary");
    this->calculatedBoundary = getMeshBoundary(this->mesh, "calculatedBoundary");
    this->defaultBoundary = getMeshBoundary(this->mesh, "defaultBoundary");
    this->init();
}

void Mesh::init () {
    for (auto& patch: this->boundary) {
        string patchID = patch.first;
        auto& patchInfo = patch.second;
        integer startFace = stoi(patchInfo.at("startFace"));
        integer nFaces = stoi(patchInfo.at("nFaces"));
        this->boundaryFaces[patchID] = make_pair(startFace, nFaces);
    }
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
void getMeshArray(PyObject *mesh, const string attr, arrType<dtype>& tmp) {
    PyArrayObject *array = (PyArrayObject*) PyObject_GetAttrString(mesh, attr.c_str());
    //cout << attr << " " << PyArray_DESCR(array)->elsize << endl;
    getArray(array, tmp);
}


template <typename dtype>
void getArray(PyArrayObject *array, arrType<dtype> & tmp) {
    assert(array);
    int nDims = PyArray_NDIM(array);
    npy_intp* dims = PyArray_DIMS(array);
    int rows = dims[1];
    int cols = dims[0];
    if (nDims == 1) {
        rows = 1;
    }
    dtype *data = (dtype *) PyArray_DATA(array);
    //cout << rows << " " << cols << endl;
    integer shape[NDIMS] = {cols, rows, 1, 1};
    arrType<dtype> result(shape, data);
    tmp = result;
    //Py_DECREF(array);
}

template <typename dtype>
PyObject* putArray(arrType<dtype> &tmp) {
    npy_intp shape[2] = {tmp.shape[0], tmp.shape[1]};
    #ifdef ADIFF 
        uscalar* data = tmp.adGet();
    #else
        uscalar* data = tmp.data;
    #endif
    return PyArray_SimpleNewFromData(2, shape, NPY_DOUBLE, data);
}

Boundary getMeshBoundary(PyObject *mesh, const string attr) {
    PyObject *dict = PyObject_GetAttrString(mesh, attr.c_str());
    return getBoundary(dict);
}

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
            else {
            }
            boundary[ckey][ckey2] = cvalue;
        }
    }
    return boundary;
}
