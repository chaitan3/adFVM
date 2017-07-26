#include "interface.hpp"

long long mil = 0;

long long current_timestamp() {
    struct timeval te; 
    gettimeofday(&te, NULL); // get current time
    long long milliseconds = te.tv_sec*1000LL + te.tv_usec/1000; // caculate milliseconds
     //printf("milliseconds: %lld\n", milliseconds);
         return milliseconds;
     }

Mesh *meshp = NULL;
Mesh *meshap = NULL;

#include "density.hpp"
#include "code.hpp"
#ifdef MATOP
    #include "matop.hpp"
    Matop *matop;
#endif


#define initFunc initinterface
#define modName "interface"

template <typename dtype, integer shape1, integer shape2>
void getMeshArray(PyObject *mesh, const string attr, arrType<dtype, shape1, shape2>& tmp) {
    PyArrayObject *array = (PyArrayObject*) PyObject_GetAttrString(mesh, attr.c_str());
    //cout << attr << " " << array << endl;
    //cout << attr << " " << PyArray_DESCR(array)->elsize << endl;
    assert (array != NULL);
    getArray(array, tmp);
    Py_DECREF(array);
}



PyObject* initSolver(PyObject *self, PyObject *args) {

    PyObject *meshObject = PyTuple_GetItem(args, 0);
    Py_INCREF(meshObject);

    meshp = new Mesh(meshObject);
    //meshap = new Mesh(*meshp);
    //rcf = new RCF();
    //cout << "Initialized mesh" << endl;
    //rcf = new RCF();
    //rcf->setMesh(mesh);

    if (PyTuple_Size(args) == 1) {
        return Py_None;
    }

    for (integer i = 1; i < 4; i++) {
        PyObject *boundaryObject = PyTuple_GetItem(args, i);
        Py_INCREF(boundaryObject);
        //rcf->boundaries[i-1] = getBoundary(boundaryObject);
    }
    //cout << "Initialized boundary" << endl;
    PyObject *dict = PyTuple_GetItem(args, 4);
    // mu 
    // riemann solver, face reconstructor support?
    PyObject *key, *value;
    Py_ssize_t pos = 0;

    #ifdef MATOP
        integer argc = 0;
        PetscInitialize(&argc, NULL, NULL, NULL);
        matop = new Matop(rcf);
    #endif

    while (PyDict_Next(dict, &pos, &key, &value)) {
        string ckey = PyString_AsString(key);
        if (ckey == "CFL") {
            //rcf->CFL = PyFloat_AsDouble(value);
        } else if (ckey == "objectivePLInfo") {
            if (value != Py_None) {
                //getDict(value, rcf->objectivePLInfo);
            }
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

    //for (int i = 0; i < nStages; i++) {
    //    rhos[i] = NULL;
    //    rhoUs[i] = NULL;
    //    rhoEs[i] = NULL;
    //}
    

    Py_INCREF(Py_None);
    return Py_None;
}

//static PyObject* forwardSolver(PyObject *self, PyObject *args) {
//    const Mesh& mesh = *meshp;
//
//    //cout << "forward 1" << endl;
//    PyObject *rhoObject, *rhoUObject, *rhoEObject;
//    PyObject *rhoSObject, *rhoUSObject, *rhoESObject;
//    scalar t, dt;
//    PyArg_ParseTuple(args, "OOOOOOdd", &rhoObject, &rhoUObject, &rhoEObject, &rhoSObject, &rhoUSObject, &rhoESObject, &dt, &t);
//
//    vec rho, rhoE;
//    mat rhoU;
//    getArray((PyArrayObject *)rhoObject, rho);
//    getArray((PyArrayObject *)rhoUObject, rhoU);
//    getArray((PyArrayObject *)rhoEObject, rhoE);
//    //cout << "forward 2" << endl;
//
//    vec rhoS, rhoES;
//    mat rhoUS;
//    getArray((PyArrayObject *)rhoSObject, rhoS);
//    getArray((PyArrayObject *)rhoUSObject, rhoUS);
//    getArray((PyArrayObject *)rhoESObject, rhoES);
//
//    //cout << "forward 3" << endl;
//    //const Mesh& mesh = *(rcf->mesh);
//    vec rhoN(mesh.nInternalCells, true);
//    mat rhoUN(mesh.nInternalCells, true);
//    vec rhoEN(mesh.nInternalCells, true);
//    scalar obj, dtc;
//    tie(obj, dtc) = timeIntegrator(rho, rhoU, rhoE, rhoN, rhoUN, rhoEN, t, dt);
//    //cout << "forward 4" << endl;
//    
//    PyObject *rhoNObject, *rhoUNObject, *rhoENObject;
//    rhoNObject = putArray(rhoN);
//    rhoUNObject = putArray(rhoUN);
//    rhoENObject = putArray(rhoEN);
//    timeIntegrator_exit();
//    //cout << "forward 5" << endl;
//    
//    return Py_BuildValue("(NNNdd)", rhoNObject, rhoUNObject, rhoENObject, obj, dtc);
//}
//
//static PyObject* backwardSolver(PyObject *self, PyObject *args) {
//
//    //cout << "forward 1" << endl;
//    PyArrayObject *rhoObject, *rhoUObject, *rhoEObject;
//    PyArrayObject *rhoSObject, *rhoUSObject, *rhoESObject;
//    PyObject *rhoaObject, *rhoUaObject, *rhoEaObject;
//    scalar t, dt;
//    integer source;
//    PyArg_ParseTuple(args, "OOOOOOOOOddi", &rhoObject, &rhoUObject, &rhoEObject, &rhoSObject, &rhoUSObject, &rhoESObject, &rhoaObject, &rhoUaObject, &rhoEaObject, &dt, &t, &source);
//
//    const Mesh& mesh = *meshp;
//
//    meshap = new Mesh(*meshp);
//    Mesh& meshAdj = *meshap;
//    meshAdj.reset();
//
//    vec rho, rhoE;
//    mat rhoU;
//    getArray((PyArrayObject *)rhoObject, rho);
//    getArray((PyArrayObject *)rhoUObject, rhoU);
//    getArray((PyArrayObject *)rhoEObject, rhoE);
//    vec rhoa(mesh.nInternalCells);
//    vec rhoEa(mesh.nInternalCells);
//    mat rhoUa(mesh.nInternalCells);
//    vec rhoaO, rhoEaO;
//    mat rhoUaO;
//    getArray((PyArrayObject *)rhoaObject, rhoaO);
//    getArray((PyArrayObject *)rhoUaObject, rhoUaO);
//    getArray((PyArrayObject *)rhoEaObject, rhoEaO);
//
//    //vec rhoS(mesh.nInternalCells), rhoES(mesh.nInternalCells);
//    //mat rhoUS(mesh.nInternalCells);
//    //rhoS.adCopy((uscalar*)PyArray_DATA(rhoSObject));
//    //rhoUS.adCopy((uscalar*)PyArray_DATA(rhoUSObject));
//    //rhoES.adCopy((uscalar*)PyArray_DATA(rhoESObject));
//    //rcf -> rhoS = &rhoS;
//    //rcf -> rhoUS = &rhoUS;
//    //rcf -> rhoES = &rhoES;
//
//    vec rhoN(mesh.nInternalCells, true);
//    mat rhoUN(mesh.nInternalCells, true);
//    vec rhoEN(mesh.nInternalCells, true);
//    scalar obj, dtc;
//    tie(obj, dtc) = timeIntegrator(rho, rhoU, rhoE, rhoN, rhoUN, rhoEN, t, dt);
//
//    //cout << "forward 1" << endl;
//    for (integer i = 0; i < mesh.nInternalCells; i++) {
//        scalar v = mesh.volumes(i);
//        rhoa(i) = rhoaO(i)*v;
//        for (integer j = 0; j < 3; j++) {
//            rhoUa(i, j) = rhoUaO(i, j)*v;
//        }
//        rhoEa(i) = rhoEaO(i)*v;
//    }
//    vec rhoaN(mesh.nInternalCells, true);
//    mat rhoUaN(mesh.nInternalCells, true);
//    vec rhoEaN(mesh.nInternalCells, true);
//    auto res = timeIntegrator_grad(rho, rhoU, rhoE, rhoa, rhoUa, rhoEa, rhoaN, rhoUaN, rhoEaN, t, dt);
//    for (integer i = 0; i < mesh.nInternalCells; i++) {
//        scalar v = mesh.volumes(i);
//        rhoaN(i) = rhoaN(i)/v;
//        for (integer j = 0; j < 3; j++) {
//            rhoUaN(i, j) = rhoUaN(i, j)/v;
//        }
//        rhoEaN(i) = rhoEaN(i)/v;
//    }
//    
//    //cout << "forward 2" << endl;
//
//    timeIntegrator_exit();
//    //cout << "forward 3" << endl;
//
//    //cout << "forward 4" << endl;
//    //
//    //scalar adjoint = 0.;
//    //for (integer i = 0; i < mesh.nInternalCells; i++) {
//        //uscalar v = mesh.volumes(i).value();
//        //adjoint += rhoN(i)*rhoa(i)*v;
//        //for (integer j = 0; j < 3; j++) {
//            //adjoint += rhoUN(i, j)*rhoUa(i, j)*v;
//        //}
//        //adjoint += rhoEN(i)*rhoEa(i)*v;
//    //}
//    //tape.registerOutput(adjoint);
//    //tape.registerOutput(objective);
//    //tape.setPassive();
//
//    //objective.setGradient(1.0);
//    //tape.evaluate();
//    //vec rhoaN(mesh.nInternalCells);
//    //mat rhoUaN(mesh.nInternalCells);
//    //vec rhoEaN(mesh.nInternalCells);
//    //rhoaN.adGetGrad(rho);
//    //rhoUaN.adGetGrad(rhoU);
//    //rhoEaN.adGetGrad(rhoE);
//    //tape.clearAdjoints();
//
//    //adjoint.setGradient(1.0);
//    //tape.evaluate();
//    //for (integer i = 0; i < mesh.nInternalCells; i++) {
//        //uscalar v = mesh.volumes(i).value();
//        //rhoaN(i) = rhoaN(i)/v + rho(i).getGradient()/v;
//        //for (integer j = 0; j < 3; j++) {
//            //rhoUaN(i, j) = rhoUaN(i, j)/v + rhoU(i, j).getGradient()/v;
//        //}
//        //rhoEaN(i) = rhoEaN(i)/v + rhoE(i).getGradient()/v;
//    //}
//    PyObject *rhoaNObject, *rhoUaNObject, *rhoEaNObject;
//    rhoaNObject = putArray(rhoaN);
//    rhoUaNObject = putArray(rhoUaN);
//    rhoEaNObject = putArray(rhoEaN);
//    if (source) {
//        vec rhoSa(mesh.nInternalCells, true);
//        mat rhoUSa(mesh.nInternalCells, true);
//        vec rhoESa(mesh.nInternalCells, true);
//        PyObject *rhoSaObject, *rhoUSaObject, *rhoESaObject;
//        rhoSaObject = putArray(rhoSa);
//        rhoUSaObject = putArray(rhoUSa);
//        rhoESaObject = putArray(rhoESa);
//        delete meshap;
//        return Py_BuildValue("(NNNNNN)", rhoaNObject, rhoUaNObject, rhoEaNObject, rhoSaObject, rhoUSaObject, rhoESaObject);
//    } else {
//        PyObject *areasObject = putArray(meshAdj.areas);
//        PyObject *volumesLObject = putArray(meshAdj.volumesL);
//        PyObject *volumesRObject = putArray(meshAdj.volumesR);
//        PyObject *normalsObject = putArray(meshAdj.normals);
//        PyObject *weightsObject = putArray(meshAdj.weights);
//        PyObject *deltasObject = putArray(meshAdj.deltas);
//        PyObject *linearWeightsObject = putArray(meshAdj.linearWeights);
//        PyObject *quadraticWeightsObject = putArray(meshAdj.quadraticWeights);
//        delete meshap;
//        return Py_BuildValue("(NNNNNNNNNNN)", rhoaNObject, rhoUaNObject, rhoEaNObject, areasObject, volumesLObject, volumesRObject, weightsObject, \
//                deltasObject, normalsObject, linearWeightsObject, quadraticWeightsObject);
//
//    }
//}
//
//static PyObject* ghost(PyObject *self, PyObject *args) {
//    const Mesh& mesh = *meshp;
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
//    //const Mesh& mesh = *(rcf->mesh);
//    mat U(mesh.nCells, true);
//    vec T(mesh.nCells, true);
//    vec p(mesh.nCells, true);
//    Function_primitive(mesh.nInternalCells, &rho(0), &rhoU(0), &rhoE(0), &U(0), &T(0), &p(0));
//
//    rcf->boundaryUPT(U, T, p);
//    rcf->boundaryEnd();
//    //cout << "forward 3" << endl;
//    //
//    vec rhoN(mesh.nCells, true);
//    mat rhoUN(mesh.nCells, true);
//    vec rhoEN(mesh.nCells, true);
//    Function_conservative(mesh.nCells, &U(0), &T(0), &p(0), &rhoN(0), &rhoUN(0), &rhoEN(0));
//    //cout << "forward 4" << endl;
//    
//    PyObject *rhoNObject, *rhoUNObject, *rhoENObject;
//    rhoNObject = putArray(rhoN);
//    rhoUNObject = putArray(rhoUN);
//    rhoENObject = putArray(rhoEN);
//    //cout << "forward 5" << endl;
//    
//    return Py_BuildValue("(NNN)", rhoNObject, rhoUNObject, rhoENObject);
//}
//static PyObject* ghost_default(PyObject *self, PyObject *args) {
//
//    //cout << "forward 1" << endl;
//    const Mesh& mesh = *meshp;
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
//    rcf->boundaryInit(0);
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

PyObject* viscosity(PyObject *self, PyObject *args) {

    //cout << "forward 1" << endl;
    PyObject *uObject, *rhoObject, *rhoUObject, *rhoEObject;
    bool report;
    scalar dt, scaling;
    PyArg_ParseTuple(args, "OOOOddb", &uObject, &rhoObject, &rhoUObject, &rhoEObject, &scaling, &dt, &report);

    arrType<scalar, 5> u;
    vec rho, rhoE;
    mat rhoU;
    getArray((PyArrayObject *)uObject, u);
    getArray((PyArrayObject *)rhoObject, rho);
    getArray((PyArrayObject *)rhoUObject, rhoU);
    getArray((PyArrayObject *)rhoEObject, rhoE);
    const Mesh& mesh = *meshp;

    vec M_2norm(mesh.nCells);
    vec DT(mesh.nFaces);

    //matop->viscosity(rho, rhoU, rhoE, M_2norm, DT, scaling, report);
    //return putArray(M_2norm);

    arrType<scalar, 5> un(mesh.nInternalCells);
    #ifdef MATOP
        matop->viscosity(rho, rhoU, rhoE, M_2norm, DT, scaling, report);
        matop->heat_equation(rcf, u, DT, dt, un);
    #endif
    
    return putArray(un);
}

PyObject* finalSolver(PyObject *self, PyObject *args) {
    //delete rcf;
    delete meshp;
    #ifdef MATOP
        PetscFinalize();
        delete matop;
    #endif
}

extern PyMethodDef Methods[];
PyMODINIT_FUNC
initFunc(void)
{
    PyObject *m;

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
    getMeshArray(this->mesh, "faceCentres", this->faceCentres);
    getMeshArray(this->mesh, "areas", this->areas);
    getMeshArray(this->mesh, "cellFaces", this->cellFaces);
    getMeshArray(this->mesh, "cellNeighboursMatOp", this->cellNeighbours);
    //getMeshArray(this->mesh, "cellCentres", this->cellCentres);
    getMeshArray(this->mesh, "volumes", this->volumes);
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

Mesh::Mesh(const Mesh& mesh) {
    this->areas = move(vec(mesh.nFaces, true));
    this->normals = move(mat(mesh.nFaces));
    this->volumesL = move(vec(mesh.nFaces));
    this->volumesR = move(vec(mesh.nInternalFaces));
    this->deltas = move(vec(mesh.nFaces));
    this->weights = move(vec(mesh.nFaces));
    this->linearWeights = move(arrType<scalar, 2>(mesh.nFaces));
    this->quadraticWeights = move(arrType<scalar, 2, 3>(mesh.nFaces));
}

void Mesh::reset() {
    this->areas.zero();
    this->normals.zero();
    this->volumesL.zero();
    this->volumesR.zero();
    this->deltas.zero();
    this->weights.zero();
    this->linearWeights.zero();
    this->quadraticWeights.zero();
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

#ifdef MATOP
    extern "C"{
    void dsyev_( char* jobz, char* uplo, int* n, double* a, int* lda,
        double* w, double* work, int* lwork, int* info );
    }

    scalar getMaxEigenvalue(arrType<double, 5, 5>& phi, vec& eigPhi) {
        char jobz = 'N';
        char uplo = 'U';
        int n = 5;
        int lda = 5;
        int lwork = 3*n-1;
        double work[lwork];
        int info;
        double w[5];

        for (int i = 0; i < phi.shape; i++) {
            dsyev_(&jobz, &uplo, &n, &phi(i), &lda, w, work, &lwork, &info);
            assert(info == 0);
            eigPhi(i) = w[4];
        }
        return 0;
    }
#endif

static MPI_Request* mpi_req;
static integer mpi_reqIndex;
static integer mpi_reqField = 0;
static scalar* mpi_reqBuf[3];

template <typename dtype, integer shape1, integer shape2>
void Function_mpi(std::vector<arrType<dtype, shape1, shape2>*> phiP) {
    arrType<dtype, shape1, shape2>& phi = *(phiP[0]);
    const Mesh& mesh = *meshp;
    //MPI_Barrier(MPI_COMM_WORLD);

    dtype* phiBuf = NULL;
    integer reqPos = 0;
    if (mesh.nRemotePatches > 0) {
        reqPos = mpi_reqIndex/(2*mesh.nRemotePatches);
        phiBuf = new dtype[(mesh.nCells-mesh.nLocalCells)*shape1*shape2];
        mpi_reqBuf[reqPos] = phiBuf;
    }

    for (auto& patch: mesh.boundary) {
        string patchType = patch.second.at("type");
        if (patchType == "processor" || patchType == "processorCyclic") {
            string patchID = patch.first;
            const map<string, string>& patchInfo = mesh.boundary.at(patchID);

            integer startFace, nFaces;
            tie(startFace, nFaces) = mesh.boundaryFaces.at(patch.first);
            integer cellStartFace = mesh.nInternalCells + startFace - mesh.nInternalFaces;
                //cout << "hello " << patchID << endl;
                integer bufStartFace = cellStartFace - mesh.nLocalCells;
                integer size = nFaces*shape1*shape2;
                integer dest = stoi(patchInfo.at("neighbProcNo"));
                for (integer i = 0; i < nFaces; i++) {
                    integer p = mesh.owner(startFace + i);
                    integer b = bufStartFace + i;
                    for (integer j = 0; j < shape1; j++) {
                        for (integer k = 0; k < shape2; k++) {
                            phiBuf[b*shape1*shape2 + j*shape2 + k] = phi(p, j, k);
                        }
                    }
                }
                MPI_Request *req = mpi_req;
                integer tag = mpi_reqField*100 + mesh.tags.at(patchID);
                //cout << patchID << " " << tag << endl;
                MPI_Isend(&phiBuf[bufStartFace*shape1*shape2], size, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD, &req[mpi_reqIndex]);
                MPI_Irecv(&phi(cellStartFace), size, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD, &req[mpi_reqIndex+1]);
                mpi_reqIndex += 2;
        }
    }
    mpi_reqField = (mpi_reqField + 1) % 100;
}
template void Function_mpi<>(std::vector<arrType<scalar, 1, 1>*> phiP);
template void Function_mpi<>(std::vector<arrType<scalar, 1, 3>*> phiP);
template void Function_mpi<>(std::vector<arrType<scalar, 3, 1>*> phiP);
template void Function_mpi<>(std::vector<arrType<scalar, 3, 3>*> phiP);

void Function_mpi_allreduce(std::vector<vec*> vals) {
    integer n = vals.size()/2;
    vec in(n, true);
    vec out(n, true);
    for (integer i = 0; i < n; i++) {
        in(i) = (*vals[i])(0);
    }
    MPI_Allreduce(&in(0), &out(0), n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for (integer i = 0; i < n; i++) {
        (*vals[i + n])(0) = out(i);
    }
}

void Function_mpi_init() {
    const Mesh& mesh = *meshp;
    mpi_reqIndex = 0;
    if (mesh.nRemotePatches > 0) {
        //MPI_Barrier(MPI_COMM_WORLD);
        mpi_req = new MPI_Request[2*3*mesh.nRemotePatches];
    }
}

void Function_mpi_end() {
    const Mesh& mesh = *meshp;
    if (mesh.nRemotePatches > 0) {
        MPI_Waitall(mpi_reqIndex, (mpi_req), MPI_STATUSES_IGNORE);
        delete[] mpi_req;
        //MPI_Barrier(MPI_COMM_WORLD);
        for (integer i = 0; i < mpi_reqIndex/(2*mesh.nRemotePatches); i++) {
            delete[] mpi_reqBuf[i];
        }
    }
}

