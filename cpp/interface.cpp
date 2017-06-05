#include "interface.hpp"
#include "timestep.hpp"
#include "density.hpp"
#include "objective.hpp"
#ifdef MATOP
    #include "matop.hpp"
    Matop* matop;
#endif

RCF* rcf;
tuple<scalar, scalar> (*timeIntegrator)(RCF*, const vec&, const mat&, const vec&, vec&, mat&, vec&, scalar, scalar) = SSPRK;

template <typename dtype, integer shape1, integer shape2>
void getMeshArray(PyObject *mesh, const string attr, arrType<dtype, shape1, shape2>& tmp) {
    //cout << attr << endl;
    PyArrayObject *array = (PyArrayObject*) PyObject_GetAttrString(mesh, attr.c_str());
    //cout << attr << " " << PyArray_DESCR(array)->elsize << endl;
    getArray(array, tmp);
    //Py_DECREF(array);
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
    dtype *data = (dtype *) PyArray_DATA(array);
    //cout << rows << " " << cols << endl;
    arrType<dtype, shape1, shape2> result(dims[0], data);
    tmp = result;
}

template <typename dtype, integer shape1>
PyObject* putArray(arrType<dtype, shape1> &tmp) {
    npy_intp shape[2] = {tmp.shape, shape1};
    uscalar* data = tmp.data;
    tmp.ownData = false;
    PyObject* array = PyArray_SimpleNewFromData(2, shape, NPY_DOUBLE, data);
    PyArray_ENABLEFLAGS((PyArrayObject*)array, NPY_ARRAY_OWNDATA);
    return array;
}

static PyObject* initSolver(PyObject *self, PyObject *args) {

    PyObject *meshObject = PyTuple_GetItem(args, 0);
    Py_INCREF(meshObject);

    Mesh *mesh = new Mesh(meshObject);
    rcf = new RCF();
    rcf->setMesh(mesh);

    #ifdef MATOP
    #ifdef ADIFF
        integer argc = 0;
        PetscInitialize(&argc, NULL, NULL, NULL);
        matop = new Matop(rcf);
    #endif
    #endif

    if (PyTuple_Size(args) == 1) {
        return Py_None;
    }

    for (integer i = 1; i < 4; i++) {
        PyObject *boundaryObject = PyTuple_GetItem(args, i);
        Py_INCREF(boundaryObject);
        rcf->boundaries[i-1] = getBoundary(boundaryObject);
    }
    PyObject *dict = PyTuple_GetItem(args, 4);
    // mu 
    // riemann solver, face reconstructor support?
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(dict, &pos, &key, &value)) {
        string ckey = PyString_AsString(key);
        if (ckey == "Cp") {
            rcf->Cp = PyFloat_AsDouble(value);
            rcf->Cv = rcf->Cp/rcf->gamma;
        } else if (ckey == "CFL") {
            rcf->CFL = PyFloat_AsDouble(value);
        } else if (ckey == "timeIntegrator") {
            string cvalue = PyString_AsString(value);
            if (cvalue == "euler") {
                timeIntegrator = euler;
            } else if (cvalue == "SSPRK") {
                timeIntegrator = SSPRK;
            }
        } else if (ckey == "mu") {
            if (value == Py_None) {
                rcf->mu = &RCF::sutherland;
            } else {
                rcf->muC = PyFloat_AsDouble(value);
                rcf->mu = &RCF::constantMu;
            }
        } else if (ckey == "objective") {
            if (value == Py_None) {
                rcf->objective = objectiveNone;
            } else {
                string cvalue = PyString_AsString(value);
                if (cvalue == "drag") {
                    rcf->objective = objectiveDrag;
                }
                if (cvalue == "pressureLoss") {
                    rcf->objective = objectivePressureLoss;
                }
            }
        } else if (ckey == "objectiveDragInfo") {
            if (value != Py_None) {
                string cvalue = PyString_AsString(value);
                rcf->objectiveDragInfo = cvalue;
            }
        } else if (ckey == "objectivePLInfo") {
            if (value != Py_None) {
                getDict(value, rcf->objectivePLInfo);
            }
        }
    }


    Py_INCREF(Py_None);
    return Py_None;
}

#ifdef ADIFF
    #define initFunc initadFVMcpp_ad
    #define modName "adFVMcpp_ad"
    auto& tape = scalar::getGlobalTape();    

    static PyObject* forwardSolver(PyObject *self, PyObject *args) {

        //cout << "forward 1" << endl;
        PyArrayObject *rhoObject, *rhoUObject, *rhoEObject;
        PyArrayObject *rhoSObject, *rhoUSObject, *rhoESObject;
        PyObject *rhoaObject, *rhoUaObject, *rhoEaObject;
        uscalar t, dt;
        integer nSteps;
        PyArg_ParseTuple(args, "OOOOOOOOOddi", &rhoObject, &rhoUObject, &rhoEObject, &rhoSObject, &rhoUSObject, &rhoESObject, &rhoaObject, &rhoUaObject, &rhoEaObject, &dt, &t, &nSteps);

        const Mesh& mesh = *(rcf->mesh);
        vec rho(mesh.nInternalCells), rhoE(mesh.nInternalCells);
        mat rhoU(mesh.nInternalCells);
        //assert(PyArray_IS_C_CONTIGUOUS(rhoObject));
        //assert(PyArray_IS_C_CONTIGUOUS(rhoObject));
        //assert(PyArray_IS_C_CONTIGUOUS(rhoObject));
        rho.adCopy((uscalar*)PyArray_DATA(rhoObject));
        rhoU.adCopy((uscalar*)PyArray_DATA(rhoUObject));
        rhoE.adCopy((uscalar*)PyArray_DATA(rhoEObject));
        //getArray((PyArrayObject *)rhoObject, rho);
        //getArray((PyArrayObject *)rhoUObject, rhoU);
        //getArray((PyArrayObject *)rhoEObject, rhoE);
        //vec rhoa, rhoEa;
        //mat rhoUa;
        uvec rhoa, rhoEa;
        umat rhoUa;
        getArray((PyArrayObject *)rhoaObject, rhoa);
        getArray((PyArrayObject *)rhoUaObject, rhoUa);
        getArray((PyArrayObject *)rhoEaObject, rhoEa);

        vec rhoS(mesh.nInternalCells), rhoES(mesh.nInternalCells);
        mat rhoUS(mesh.nInternalCells);
        rhoS.adCopy((uscalar*)PyArray_DATA(rhoSObject));
        rhoUS.adCopy((uscalar*)PyArray_DATA(rhoUSObject));
        rhoES.adCopy((uscalar*)PyArray_DATA(rhoESObject));
        rcf -> rhoS = &rhoS;
        rcf -> rhoUS = &rhoUS;
        rcf -> rhoES = &rhoES;

        //cout << "forward 2" << endl;

        //cout << "forward 3" << endl;

        tape.setActive();
        rho.adInit(tape);
        rhoU.adInit(tape);
        rhoE.adInit(tape);
        rhoS.adInit(tape);
        rhoUS.adInit(tape);
        rhoES.adInit(tape);

        vec rhoN(mesh.nInternalCells);
        mat rhoUN(mesh.nInternalCells);
        vec rhoEN(mesh.nInternalCells);

        scalar objective, dtc;
        tie(objective, dtc) = timeIntegrator(rcf, rho, rhoU, rhoE, rhoN, rhoUN, rhoEN, t, dt);


        //mat U(mesh.nCells);
        //vec T(mesh.nCells);
        //vec p(mesh.nCells);
        //for (integer i = 0; i < mesh.nInternalCells; i++) {
        //    rcf->primitive(rho(i), &rhoU(i), rhoE(i), &U(i), T(i), p(i));
        //}
        ////cout << "c++: equation 2" << endl;

        //rcf->U = &U;
        //rcf->T = &T;
        //rcf->p = &p;
        //rcf->boundary(rcf->boundaries[0], U);
        //rcf->boundary(rcf->boundaries[1], T);
        //rcf->boundary(rcf->boundaries[2], p);
        //objective = rcf->objective(rcf, U, T, p);

        //cout << "forward 4" << endl;
        //
        scalar adjoint = 0.;
        for (integer i = 0; i < mesh.nInternalCells; i++) {
            uscalar v = mesh.volumes(i);
            adjoint += rhoN(i)*rhoa(i)*v;
            for (integer j = 0; j < 3; j++) {
                adjoint += rhoUN(i, j)*rhoUa(i, j)*v;
            }
            adjoint += rhoEN(i)*rhoEa(i)*v;
        }
        tape.registerOutput(adjoint);
        tape.registerOutput(objective);
        tape.setPassive();

        adjoint.setGradient(1.0);
        tape.evaluate();
        uvec rhoaN(mesh.nInternalCells);
        umat rhoUaN(mesh.nInternalCells);
        uvec rhoEaN(mesh.nInternalCells);
        rhoaN.adGetGrad(rho);
        rhoUaN.adGetGrad(rhoU);
        rhoEaN.adGetGrad(rhoE);
        uvec rhoSa(mesh.nInternalCells);
        umat rhoUSa(mesh.nInternalCells);
        uvec rhoESa(mesh.nInternalCells);
        rhoSa.adGetGrad(rhoS);
        rhoUSa.adGetGrad(rhoUS);
        rhoESa.adGetGrad(rhoES);

        tape.clearAdjoints();
        objective.setGradient(1.0);
        tape.evaluate();
        for (integer i = 0; i < mesh.nInternalCells; i++) {
            uscalar v = mesh.volumes(i);
            //rhoaN(i) = rhoaN(i)/v + rho(i).getGradient()/(v*nSteps);
            //for (integer j = 0; j < 3; j++) {
            //    rhoUaN(i, j) = rhoUaN(i, j)/v + rhoU(i, j).getGradient()/(v*nSteps);
            //}
            //rhoEaN(i) = rhoEaN(i)/v + rhoE(i).getGradient()/(v*nSteps);
            rhoaN(i) = rhoaN(i)/v + rho(i).getGradient()/v;
            for (integer j = 0; j < 3; j++) {
                rhoUaN(i, j) = rhoUaN(i, j)/v + rhoU(i, j).getGradient()/v;
            }
            rhoEaN(i) = rhoEaN(i)/v + rhoE(i).getGradient()/v;

        }
        tape.reset();
        
        //cout << "evaluated tape" << endl;

        PyObject *rhoaNObject, *rhoUaNObject, *rhoEaNObject;
        rhoaNObject = putArray(rhoaN);
        rhoUaNObject = putArray(rhoUaN);
        rhoEaNObject = putArray(rhoEaN);
        PyObject *rhoSaObject, *rhoUSaObject, *rhoESaObject;
        rhoSaObject = putArray(rhoSa);
        rhoUSaObject = putArray(rhoUSa);
        rhoESaObject = putArray(rhoESa);
        //cout << "forward 5" << endl;
        
        return Py_BuildValue("(NNNNNN)", rhoaNObject, rhoUaNObject, rhoEaNObject, rhoSaObject, rhoUSaObject, rhoESaObject);
    }
    
    
#else
    #define initFunc initadFVMcpp
    #define modName "adFVMcpp"

    static PyObject* forwardSolver(PyObject *self, PyObject *args) {

        //cout << "forward 1" << endl;
        PyObject *rhoObject, *rhoUObject, *rhoEObject;
        PyObject *rhoSObject, *rhoUSObject, *rhoESObject;
        uscalar t, dt;
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
        rcf -> rhoS = &rhoS;
        rcf -> rhoUS = &rhoUS;
        rcf -> rhoES = &rhoES;

        //cout << "forward 3" << endl;
        const Mesh& mesh = *(rcf->mesh);
        vec rhoN(mesh.nInternalCells);
        mat rhoUN(mesh.nInternalCells);
        vec rhoEN(mesh.nInternalCells);
        scalar objective, dtc;
        tie(objective, dtc) = timeIntegrator(rcf, rho, rhoU, rhoE, rhoN, rhoUN, rhoEN, t, dt);
        //cout << "forward 4" << endl;
        
        PyObject *rhoNObject, *rhoUNObject, *rhoENObject;
        rhoNObject = putArray(rhoN);
        rhoUNObject = putArray(rhoUN);
        rhoENObject = putArray(rhoEN);
        //cout << "forward 5" << endl;
        
        return Py_BuildValue("(NNNdd)", rhoNObject, rhoUNObject, rhoENObject, objective, dtc);
    }
    static PyObject* ghost(PyObject *self, PyObject *args) {

        //cout << "forward 1" << endl;
        PyObject *rhoObject, *rhoUObject, *rhoEObject;
        PyArg_ParseTuple(args, "OOO", &rhoObject, &rhoUObject, &rhoEObject);

        vec rho, rhoE;
        mat rhoU;
        getArray((PyArrayObject *)rhoObject, rho);
        getArray((PyArrayObject *)rhoUObject, rhoU);
        getArray((PyArrayObject *)rhoEObject, rhoE);
        //cout << "forward 2" << endl;
        //
        const Mesh& mesh = *(rcf->mesh);
        mat U(mesh.nCells);
        vec T(mesh.nCells);
        vec p(mesh.nCells);
        for (integer i = 0; i < mesh.nInternalCells; i++) {
            rcf->primitive(rho(i), &rhoU(i), rhoE(i), &U(i), T(i), p(i));
        }
        rcf->U = &U;
        rcf->T = &T;
        rcf->p = &p;
        rcf->boundaryInit();
        rcf->boundary(rcf->boundaries[0], U);
        rcf->boundary(rcf->boundaries[1], T);
        rcf->boundary(rcf->boundaries[2], p);
        rcf->boundaryEnd();

        //cout << "forward 3" << endl;
        vec rhoN(mesh.nCells);
        mat rhoUN(mesh.nCells);
        vec rhoEN(mesh.nCells);
        for (integer i = 0; i < mesh.nCells; i++) {
            rcf->conservative(&U(i), T(i), p(i), rhoN(i), &rhoUN(i), rhoEN(i));
        }
        //cout << "forward 4" << endl;
        
        PyObject *rhoNObject, *rhoUNObject, *rhoENObject;
        rhoNObject = putArray(rhoN);
        rhoUNObject = putArray(rhoUN);
        rhoENObject = putArray(rhoEN);
        //cout << "forward 5" << endl;
        
        return Py_BuildValue("(NNN)", rhoNObject, rhoUNObject, rhoENObject);
    }
    static PyObject* ghost_default(PyObject *self, PyObject *args) {

        //cout << "forward 1" << endl;
        PyObject *rhoObject, *rhoUObject, *rhoEObject;
        PyArg_ParseTuple(args, "OOO", &rhoObject, &rhoUObject, &rhoEObject);

        uvec rho, rhoE;
        umat rhoU;
        getArray((PyArrayObject *)rhoObject, rho);
        getArray((PyArrayObject *)rhoUObject, rhoU);
        getArray((PyArrayObject *)rhoEObject, rhoE);
        //cout << "forward 2" << endl;
        //
        
        const Mesh& mesh = *(rcf->mesh);
        uvec rhoN(mesh.nCells);
        umat rhoUN(mesh.nCells);
        uvec rhoEN(mesh.nCells);

        for (integer i = 0; i < mesh.nInternalCells; i++) {
            rhoN(i) = rho(i);
            for (integer j = 0; j < 3; j++) {
                rhoUN(i, j) = rhoU(i, j);
            }
            rhoEN(i) = rhoE(i);
        }

        rcf->boundaryInit();
        rcf->boundary(mesh.defaultBoundary, rhoN);
        rcf->boundary(mesh.defaultBoundary, rhoUN);
        rcf->boundary(mesh.defaultBoundary, rhoEN);
        rcf->boundaryEnd();

        //cout << "forward 3" << endl;
        
        PyObject *rhoNObject, *rhoUNObject, *rhoENObject;
        rhoNObject = putArray(rhoN);
        rhoUNObject = putArray(rhoUN);
        rhoENObject = putArray(rhoEN);
        //cout << "forward 5" << endl;
        
        return Py_BuildValue("(NNN)", rhoNObject, rhoUNObject, rhoENObject);
    }
#endif

    static PyObject* viscosity(PyObject *self, PyObject *args) {

        //cout << "forward 1" << endl;
        PyObject *uObject, *DTObject;
        uscalar dt;
        PyArg_ParseTuple(args, "OOd", &uObject, &DTObject, &dt);

        arrType<uscalar, 5> u;
        uvec DT;
        getArray((PyArrayObject *)uObject, u);
        getArray((PyArrayObject *)DTObject, DT);
        const Mesh& mesh = *(rcf->mesh);

        arrType<uscalar, 5> un(mesh.nInternalCells);
        #ifdef MATOP
            matop->heat_equation(rcf, u, DT, dt, un);
        #endif
        
        PyObject *uNObject = putArray(un);
        return uNObject;
    }

PyMODINIT_FUNC
initFunc(void)
{
    PyObject *m;

    static PyMethodDef Methods[] = {
        {"forward",  forwardSolver, METH_VARARGS, "boo"},
        {"init",  initSolver, METH_VARARGS, "Execute a shell command."},
        #ifndef ADIFF
        {"ghost",  ghost, METH_VARARGS, "Execute a shell command."},
        {"ghost_default",  ghost_default, METH_VARARGS, "Execute a shell command."},
        #endif
        {"viscosity",  viscosity, METH_VARARGS, "Execute a shell command."},
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

    getMeshArray(this->mesh, "faces", this->faces);
    getMeshArray(this->mesh, "points", this->points);
    getMeshArray(this->mesh, "owner", this->owner);
    getMeshArray(this->mesh, "neighbour", this->neighbour);

    getMeshArray(this->mesh, "normals", this->normals);
    getMeshArray(this->mesh, "faceCentres", this->faceCentres);
    getMeshArray(this->mesh, "areas", this->areas);
    getMeshArray(this->mesh, "cellFaces", this->cellFaces);
    getMeshArray(this->mesh, "cellNeighboursMatOp", this->cellNeighbours);
    getMeshArray(this->mesh, "cellCentres", this->cellCentres);
    getMeshArray(this->mesh, "volumes", this->volumes);

    getMeshArray(this->mesh, "deltas", this->deltas);
    getMeshArray(this->mesh, "deltasUnit", this->deltasUnit);
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

map<string, integer> getTags(PyObject *mesh, const string attr) {
    PyObject *dict = PyObject_GetAttrString(mesh, attr.c_str());
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
            uscalar ivalue = PyFloat_AsDouble(value2);
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

