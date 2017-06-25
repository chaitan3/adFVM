#include "interface.hpp"

long long mil = 0;

long long current_timestamp() {
    struct timeval te; 
    gettimeofday(&te, NULL); // get current time
    long long milliseconds = te.tv_sec*1000LL + te.tv_usec/1000; // caculate milliseconds
     //printf("milliseconds: %lld\n", milliseconds);
         return milliseconds;
     }

Mesh *meshp;

#include "code.c"
#include "density.cpp"

template <typename dtype, integer shape1, integer shape2>
void getMeshArray(PyObject *mesh, const string attr, arrType<dtype, shape1, shape2>& tmp) {
    PyArrayObject *array = (PyArrayObject*) PyObject_GetAttrString(mesh, attr.c_str());
    //cout << attr << " " << array << endl;
    //cout << attr << " " << PyArray_DESCR(array)->elsize << endl;
    assert (array != NULL);
    getArray(array, tmp);
    Py_DECREF(array);
}


template <typename dtype, integer shape1, integer shape2>
void getArray(PyArrayObject *array, arrType<dtype, shape1, shape2> & tmp) {
    assert(array);
    int nDims = PyArray_NDIM(array);
    npy_intp* dims = PyArray_DIMS(array);
    if (nDims > 1) {
        assert(dims[1] == shape1);
        //cout << dims[1] << " " << shape1 << endl;
    }
    if (nDims > 2) {
        assert(dims[2] == shape2);
        //cout << dims[2] << " " << shape2 << endl;
    }
    assert(PyArray_IS_C_CONTIGUOUS(array));

    arrType<dtype, shape1, shape2> result(dims[0], (dtype *) PyArray_DATA(array));
    //cout << rows << " " << cols << endl;
    //if ((typeid(dtype) != type(uscalar)) && (typeid(dtype) != typeid(integer))) {
    tmp = result;
    result.ownData = false;
}

template <typename dtype, integer shape1>
PyObject* putArray(arrType<dtype, shape1> &tmp) {
    npy_intp shape[2] = {tmp.shape, shape1};
    scalar* data = tmp.data;
    tmp.ownData = false;
    PyObject* array = PyArray_SimpleNewFromData(2, shape, NPY_DOUBLE, data);
    PyArray_ENABLEFLAGS((PyArrayObject*)array, NPY_ARRAY_OWNDATA);
    return array;
}
template <typename dtype, integer shape1, integer shape2>
PyObject* putArray(arrType<dtype, shape1, shape2> &tmp) {
    npy_intp shape[3] = {tmp.shape, shape1, shape2};
    scalar* data = tmp.data;
    tmp.ownData = false;
    PyObject* array = PyArray_SimpleNewFromData(3, shape, NPY_DOUBLE, data);
    PyArray_ENABLEFLAGS((PyArrayObject*)array, NPY_ARRAY_OWNDATA);
    return array;
}

static PyObject* initSolver(PyObject *self, PyObject *args) {

    PyObject *meshObject = PyTuple_GetItem(args, 0);
    Py_INCREF(meshObject);

    meshp = new Mesh(meshObject);
    rcf = new RCF();
    //cout << "Initialized mesh" << endl;
    //rcf = new RCF();
    //rcf->setMesh(mesh);

    if (PyTuple_Size(args) == 1) {
        return Py_None;
    }

    for (integer i = 1; i < 4; i++) {
        PyObject *boundaryObject = PyTuple_GetItem(args, i);
        Py_INCREF(boundaryObject);
        rcf->boundaries[i-1] = getBoundary(boundaryObject);
    }
    //cout << "Initialized boundary" << endl;
    PyObject *dict = PyTuple_GetItem(args, 4);
    // mu 
    // riemann solver, face reconstructor support?
    PyObject *key, *value;
    Py_ssize_t pos = 0;

    while (PyDict_Next(dict, &pos, &key, &value)) {
        string ckey = PyString_AsString(key);
        if (ckey == "CFL") {
            rcf->CFL = PyFloat_AsDouble(value);
        } 
        //else if (ckey == "timeIntegrator") {
        //    string cvalue = PyString_AsString(value);
        //    if (cvalue == "euler") {
        //        timeIntegrator = euler;
        //    } else if (cvalue == "SSPRK") {
        //        timeIntegrator = SSPRK;
        //    }
        //} 
    }
    

    Py_INCREF(Py_None);
    return Py_None;
}

#define initFunc initinterface
#define modName "interface"

static PyObject* forwardSolver(PyObject *self, PyObject *args) {
    const Mesh& mesh = *meshp;

    //cout << "forward 1" << endl;
    PyObject *rhoObject, *rhoUObject, *rhoEObject;
    PyObject *rhoSObject, *rhoUSObject, *rhoESObject;
    scalar t, dt;
    PyArg_ParseTuple(args, "OOOOOOdd", &rhoObject, &rhoUObject, &rhoEObject, &rhoSObject, &rhoUSObject, &rhoESObject, &dt, &t);

    vec rho, rhoE;
    mat rhoU;
    getArray((PyArrayObject *)rhoObject, rho);
    getArray((PyArrayObject *)rhoUObject, rhoU);
    getArray((PyArrayObject *)rhoEObject, rhoE);
    //cout << "forward 2" << endl;

    vec rhoS, rhoES;
    mat rhoUS;
    getArray((PyArrayObject *)rhoSObject, rhoS);
    getArray((PyArrayObject *)rhoUSObject, rhoUS);
    getArray((PyArrayObject *)rhoESObject, rhoES);

    //cout << "forward 3" << endl;
    //const Mesh& mesh = *(rcf->mesh);
    vec rhoN(mesh.nInternalCells);
    mat rhoUN(mesh.nInternalCells);
    vec rhoEN(mesh.nInternalCells);
    scalar objective, dtc;
    tie(objective, dtc) = timeIntegrator(rho, rhoU, rhoE, rhoN, rhoUN, rhoEN, t, dt);
    //cout << "forward 4" << endl;
    
    PyObject *rhoNObject, *rhoUNObject, *rhoENObject;
    rhoNObject = putArray(rhoN);
    rhoUNObject = putArray(rhoUN);
    rhoENObject = putArray(rhoEN);
    //cout << "forward 5" << endl;
    
    return Py_BuildValue("(NNNdd)", rhoNObject, rhoUNObject, rhoENObject, objective, dtc);
}
//static PyObject* ghost(PyObject *self, PyObject *args) {
//
//    //cout << "forward 1" << endl;
//    //PyObject *rhoObject, *rhoUObject, *rhoEObject;
//    //PyArg_ParseTuple(args, "OOO", &rhoObject, &rhoUObject, &rhoEObject);
//
//    //vec rho, rhoE;
//    //mat rhoU;
//    //getArray((PyArrayObject *)rhoObject, rho);
//    //getArray((PyArrayObject *)rhoUObject, rhoU);
//    //getArray((PyArrayObject *)rhoEObject, rhoE);
//    ////cout << "forward 2" << endl;
//    ////
//    ////const Mesh& mesh = *(rcf->mesh);
//    //mat U(mesh.nCells);
//    //vec T(mesh.nCells);
//    //vec p(mesh.nCells);
//    //for (integer i = 0; i < mesh.nInternalCells; i++) {
//    //    //rcf->primitive(rho(i), &rhoU(i), rhoE(i), &U(i), T(i), p(i));
//    //}
//    //rcf->U = &U;
//    //rcf->T = &T;
//    //rcf->p = &p;
//    //rcf->boundaryInit();
//    //rcf->boundary(rcf->boundaries[0], U);
//    //rcf->boundary(rcf->boundaries[1], T);
//    //rcf->boundary(rcf->boundaries[2], p);
//    //rcf->boundaryEnd();
//
//    ////cout << "forward 3" << endl;
//    //vec rhoN(mesh.nCells);
//    //mat rhoUN(mesh.nCells);
//    //vec rhoEN(mesh.nCells);
//    //for (integer i = 0; i < mesh.nCells; i++) {
//    //    rcf->conservative(&U(i), T(i), p(i), rhoN(i), &rhoUN(i), rhoEN(i));
//    //}
//    ////cout << "forward 4" << endl;
//    
//    PyObject *rhoNObject, *rhoUNObject, *rhoENObject;
//    //rhoNObject = putArray(rhoN);
//    //rhoUNObject = putArray(rhoUN);
//    //rhoENObject = putArray(rhoEN);
//    //cout << "forward 5" << endl;
//    
//    return Py_BuildValue("(NNN)", rhoNObject, rhoUNObject, rhoENObject);
//}
//static PyObject* ghost_default(PyObject *self, PyObject *args) {
//
//    //cout << "forward 1" << endl;
//    PyObject *rhoObject, *rhoUObject, *rhoEObject;
//    PyArg_ParseTuple(args, "OOO", &rhoObject, &rhoUObject, &rhoEObject);
//
//    vec rho, rhoE;
//    mat rhoU;
//    getArray((PyArrayObject *)rhoObject, rho);
//    getArray((PyArrayObject *)rhoUObject, rhoU);
//    getArray((PyArrayObject *)rhoEObject, rhoE);
//    //cout << "forward 2" << endl;
//    //
//    
//    const Mesh& mesh = *(rcf->mesh);
//    vec rhoN(mesh.nCells);
//    mat rhoUN(mesh.nCells);
//    vec rhoEN(mesh.nCells);
//
//    for (integer i = 0; i < mesh.nInternalCells; i++) {
//        rhoN(i) = rho(i);
//        for (integer j = 0; j < 3; j++) {
//            rhoUN(i, j) = rhoU(i, j);
//        }
//        rhoEN(i) = rhoE(i);
//    }
//
//    rcf->boundaryInit();
//    rcf->boundary(mesh.defaultBoundary, rhoN);
//    rcf->boundary(mesh.defaultBoundary, rhoUN);
//    rcf->boundary(mesh.defaultBoundary, rhoEN);
//    rcf->boundaryEnd();
//
//    //cout << "forward 3" << endl;
//    
//    PyObject *rhoNObject, *rhoUNObject, *rhoENObject;
//    rhoNObject = putArray(rhoN);
//    rhoUNObject = putArray(rhoUN);
//    rhoENObject = putArray(rhoEN);
//    //cout << "forward 5" << endl;
//    
//    return Py_BuildValue("(NNN)", rhoNObject, rhoUNObject, rhoENObject);
//}

PyMODINIT_FUNC
initFunc(void)
{
    PyObject *m;

    static PyMethodDef Methods[] = {
        {"forward",  forwardSolver, METH_VARARGS, "boo"},
        {"init",  initSolver, METH_VARARGS, "Execute a shell command."},
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
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::cout << "Initializing C++ interface" << endl;
    }

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

    //getMeshArray(this->mesh, "faces", this->faces);
    //getMeshArray(this->mesh, "points", this->points);
    getMeshArray(this->mesh, "owner", this->owner);
    getMeshArray(this->mesh, "neighbour", this->neighbour);

    getMeshArray(this->mesh, "normals", this->normals);
    //getMeshArray(this->mesh, "faceCentres", this->faceCentres);
    getMeshArray(this->mesh, "areas", this->areas);
    //getMeshArray(this->mesh, "cellFaces", this->cellFaces);
    //getMeshArray(this->mesh, "cellNeighboursMatOp", this->cellNeighbours);
    //getMeshArray(this->mesh, "cellCentres", this->cellCentres);
    //getMeshArray(this->mesh, "volumes", this->volumes);
    getMeshArray(this->mesh, "volumesL", this->volumesL);
    getMeshArray(this->mesh, "volumesR", this->volumesR);

    getMeshArray(this->mesh, "deltas", this->deltas);
    //getMeshArray(this->mesh, "deltasUnit", this->deltasUnit);
    getMeshArray(this->mesh, "weights", this->weights);
    getMeshArray(this->mesh, "linearWeights", this->linearWeights);
    getMeshArray(this->mesh, "quadraticWeights", this->quadraticWeights);

    this->boundary = getMeshBoundary(this->mesh, "boundary");
    this->calculatedBoundary = getMeshBoundary(this->mesh, "calculatedBoundary");
    this->defaultBoundary = getMeshBoundary(this->mesh, "defaultBoundary");
    this->tags = getTags(this->mesh, "tags");
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
    MPI_Comm_size(MPI_COMM_WORLD, &this->nProcs);
}

Mesh::~Mesh () {
    Py_DECREF(this->mesh);
    Py_DECREF(this->meshClass);
    Py_DECREF(this->meshModule);
    if (Py_IsInitialized())
        Py_Finalize();
}


int getInteger(PyObject *mesh, const string attr) {
    PyObject *in = PyObject_GetAttrString(mesh, attr.c_str());
    assert (in != NULL);
    int result = (int)PyInt_AsLong(in);
    //cout << attr << " " << result << endl;
    Py_DECREF(in);
    return result;
}

string getString(PyObject *mesh, const string attr) {
    PyObject *cstring = PyObject_GetAttrString(mesh, attr.c_str());
    assert (cstring);
    string result(PyString_AsString(cstring));
    Py_DECREF(cstring);
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

void getDict(PyObject* dict, map<string, string>& cDict) {
    PyObject *key2, *value2;
    Py_ssize_t pos2 = 0;
    while (PyDict_Next(dict, &pos2, &key2, &value2)) {
        string ckey2 = PyString_AsString(key2);
        string cvalue;
        if (PyInt_Check(value2)) {
            int ivalue = (int)PyInt_AsLong(value2);
            cvalue = to_string(ivalue);
        }
        if (PyFloat_Check(value2)) {
            scalar ivalue = PyFloat_AsDouble(value2);
            cvalue = to_string(ivalue);
        }
        else if (PyString_Check(value2)) {
            cvalue = PyString_AsString(value2);
        }
        else if (PyArray_Check(value2)) {
            Py_INCREF(value2);
            PyArrayObject* val = (PyArrayObject*) value2;
            char* data = (char *) PyArray_DATA(val);
            int size = PyArray_NBYTES(val);
            cvalue = string(data, size);
        }
        cDict[ckey2] = cvalue;
    }
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

