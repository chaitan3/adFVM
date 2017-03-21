#include "interface.hpp"
#include "timestep.hpp"
#include "density.hpp"

RCF* rcf;

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

#ifdef ADIFF
    #define initFunc initadFVMcpp_ad
    #define modName "adFVMcpp_ad"
    auto& tape = codi::RealReverse::getGlobalTape();    

    static PyObject* forwardSolver(PyObject *self, PyObject *args) {

        //cout << "forward 1" << endl;
        PyObject *rhoObject, *rhoUObject, *rhoEObject;
        PyObject *rhoaObject, *rhoUaObject, *rhoEaObject;
        uscalar t, dt;
        PyArg_ParseTuple(args, "OOOOOOdd", &rhoObject, &rhoUObject, &rhoEObject, &rhoaObject, &rhoUaObject, &rhoEaObject, &dt, &t);

        arr rho, rhoU, rhoE;
        getArray((PyArrayObject *)rhoObject, rho);
        getArray((PyArrayObject *)rhoUObject, rhoU);
        getArray((PyArrayObject *)rhoEObject, rhoE);
        arr rhoa, rhoUa, rhoEa;
        getArray((PyArrayObject *)rhoaObject, rhoa);
        getArray((PyArrayObject *)rhoUaObject, rhoUa);
        getArray((PyArrayObject *)rhoEaObject, rhoEa);
        //cout << "forward 2" << endl;

        //cout << "forward 3" << endl;
        tape.reset();
        tape.setActive();
        rho.adInit(tape);
        rhoU.adInit(tape);
        rhoE.adInit(tape);

        // useless
        rhoa.adInit(tape);
        rhoUa.adInit(tape);
        rhoEa.adInit(tape);

        arr rhoN(rho.shape);
        arr rhoUN(rhoU.shape);
        arr rhoEN(rhoE.shape);
        timeStepper(rcf, rho, rhoU, rhoE, rhoN, rhoUN, rhoEN, t, dt);
        //cout << "forward 4" << endl;
        //
        scalar adjoint = 0.;
        const Mesh& mesh = *(rcf->mesh);
        for (integer i = 0; i < mesh.nInternalCells; i++) {
            scalar volume = mesh.volumes(i);
            adjoint += rhoN(i)*rhoa(i)*volume;
            for (integer j = 0; j < 3; j++) {
                adjoint += rhoUN(i, j)*rhoUa(i, j)*volume;
            }
            adjoint += rhoEN(i)*rhoEa(i)*volume;

        }
        tape.registerOutput(adjoint);
        tape.setPassive();

        adjoint.setGradient(1.0);
        tape.evaluate();
        //cout << "evaluated tape" << endl;

        uarr rhoaN(rho.shape);
        uarr rhoUaN(rhoU.shape);
        uarr rhoEaN(rhoE.shape);
        rhoaN.adGetGrad(rho);
        rhoUaN.adGetGrad(rhoU);
        rhoEaN.adGetGrad(rhoE);
        
        PyObject *rhoaNObject, *rhoUaNObject, *rhoEaNObject;
        rhoaNObject = putArray(rhoaN);
        rhoUaNObject = putArray(rhoUaN);
        rhoEaNObject = putArray(rhoEaN);
        //cout << "forward 5" << endl;
        
        return Py_BuildValue("(OOO)", rhoaNObject, rhoUaNObject, rhoEaNObject);
    }
    static PyObject* ghost(PyObject *self, PyObject *args) {

        //cout << "forward 1" << endl;
        PyObject *rhoObject, *rhoUObject, *rhoEObject;
        PyArg_ParseTuple(args, "OOO", &rhoObject, &rhoUObject, &rhoEObject);

        uarr rho, rhoU, rhoE;
        getArray((PyArrayObject *)rhoObject, rho);
        getArray((PyArrayObject *)rhoUObject, rhoU);
        getArray((PyArrayObject *)rhoEObject, rhoE);
        //cout << "forward 2" << endl;
        //
        
        const Mesh& mesh = *(rcf->mesh);
        uarr rhoN(mesh.nCells, 1);
        uarr rhoUN(mesh.nCells, 3);
        uarr rhoEN(rhoN.shape);
        rhoN.ownData = false;
        rhoUN.ownData = false;
        rhoEN.ownData = false;

        for (integer i = 0; i < mesh.nInternalCells; i++) {
            rhoN(i) = rho(i);
            for (integer j = 0; j < 3; j++) {
                rhoUN(i, j) = rhoU(i, j);
            }
            rhoEN(i) = rhoE(i);
        }

        rcf->boundary(mesh.defaultBoundary, rhoN);
        rcf->boundary(mesh.defaultBoundary, rhoUN);
        rcf->boundary(mesh.defaultBoundary, rhoEN);

        //cout << "forward 3" << endl;
        
        PyObject *rhoNObject, *rhoUNObject, *rhoENObject;
        rhoNObject = putArray(rhoN);
        rhoUNObject = putArray(rhoUN);
        rhoENObject = putArray(rhoEN);
        //cout << "forward 5" << endl;
        
        return Py_BuildValue("(OOO)", rhoNObject, rhoUNObject, rhoENObject);
    }
#else
    #define initFunc initadFVMcpp
    #define modName "adFVMcpp"

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
        arr rhoN(rho.shape);
        arr rhoUN(rhoU.shape);
        arr rhoEN(rhoE.shape);
        rhoN.ownData = false;
        rhoUN.ownData = false;
        rhoEN.ownData = false;
        timeStepper(rcf, rho, rhoU, rhoE, rhoN, rhoUN, rhoEN, t, dt);
        //cout << "forward 4" << endl;
        
        PyObject *rhoNObject, *rhoUNObject, *rhoENObject;
        rhoNObject = putArray(rhoN);
        rhoUNObject = putArray(rhoUN);
        rhoENObject = putArray(rhoEN);
        //cout << "forward 5" << endl;
        
        return Py_BuildValue("(OOO)", rhoNObject, rhoUNObject, rhoENObject);
    }
    static PyObject* ghost(PyObject *self, PyObject *args) {

        //cout << "forward 1" << endl;
        PyObject *rhoObject, *rhoUObject, *rhoEObject;
        uscalar t, dt;
        PyArg_ParseTuple(args, "OOO", &rhoObject, &rhoUObject, &rhoEObject);

        arr rho, rhoU, rhoE;
        getArray((PyArrayObject *)rhoObject, rho);
        getArray((PyArrayObject *)rhoUObject, rhoU);
        getArray((PyArrayObject *)rhoEObject, rhoE);
        //cout << "forward 2" << endl;
        //
        const Mesh& mesh = *(rcf->mesh);
        arr U(mesh.nCells, 3);
        arr T(mesh.nCells);
        arr p(T.shape);
        for (integer i = 0; i < mesh.nInternalCells; i++) {
            rcf->primitive(rho(i), &rhoU(i), rhoE(i), &U(i), T(i), p(i));
        }
        rcf->boundary(rcf->boundaries[0], U);
        rcf->boundary(rcf->boundaries[1], T);
        rcf->boundary(rcf->boundaries[2], p);

        //cout << "forward 3" << endl;
        arr rhoN(mesh.nCells, 1);
        arr rhoUN(mesh.nCells, 3);
        arr rhoEN(rhoN.shape);
        rhoN.ownData = false;
        rhoUN.ownData = false;
        rhoEN.ownData = false;
        for (integer i = 0; i < mesh.nCells; i++) {
            rcf->conservative(&U(i), T(i), p(i), rhoN(i), &rhoUN(i), rhoEN(i));
        }
        //cout << "forward 4" << endl;
        
        PyObject *rhoNObject, *rhoUNObject, *rhoENObject;
        rhoNObject = putArray(rhoN);
        rhoUNObject = putArray(rhoUN);
        rhoENObject = putArray(rhoEN);
        //cout << "forward 5" << endl;
        
        return Py_BuildValue("(OOO)", rhoNObject, rhoUNObject, rhoENObject);
    }
#endif
   
PyMODINIT_FUNC
initFunc(void)
{
    PyObject *m;

    static PyMethodDef Methods[] = {
        {"forward",  forwardSolver, METH_VARARGS, "boo"},
        {"init",  initSolver, METH_VARARGS, "Execute a shell command."},
        {"ghost",  ghost, METH_VARARGS, "Execute a shell command."},
        {NULL, NULL, 0, NULL}        /* Sentinel */
    };

    m = Py_InitModule(modName, Methods);
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
    this->nLocalPatches = getInteger(this->mesh, "nLocalPatches");
    this->nRemotePatches = getInteger(this->mesh, "nRemotePatches");

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
    uscalar* data = tmp.data;
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
            else if (ckey2[0] == "_") {
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
