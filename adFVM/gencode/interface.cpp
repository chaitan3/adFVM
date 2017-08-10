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

//#include "code.hpp"
#ifdef MATOP
    #include "matop.hpp"
    Matop *matop;
#endif


#ifdef PY3
    #define initFunc PyInit_interface
#else
    #define initFunc initinterface
#endif
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

    Py_INCREF(Py_None);
    return Py_None;
}



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
    return NULL;
}

extern PyMethodDef Methods[];

#ifdef PY3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,  /* m_base */
    modName,                 /* m_name */
    NULL,                   /* m_doc */
    -1,                     /* m_size */
    Methods            /* m_methods */
};
#endif

PyMODINIT_FUNC
initFunc(void)
{
    PyObject *m;

    #ifdef PY3
        m = PyModule_Create(&moduledef);
    #else
        m = Py_InitModule(modName, Methods);
        if (m == NULL)
            return;
    #endif
    import_array();

    //SpamError = PyErr_NewException("spam.error", NULL, NULL);
    //Py_INCREF(SpamError);
    //PyModule_AddObject(m, "error", SpamError);
    #ifdef PY3
        return m;
    #endif
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

#ifdef PY3
char* PyString_AsString(PyObject* result) {
    char *my_result;
    PyObject * temp_bytes = PyUnicode_AsEncodedString(result, "ASCII", "strict"); // Owned reference
    if (temp_bytes != NULL) {
        my_result = PyBytes_AS_STRING(temp_bytes); // Borrowed pointer
        my_result = strdup(my_result);
        Py_DECREF(temp_bytes);
        return my_result;
    } else {
        return NULL;
    }
}
#endif

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
static map<void *, void *> mpi_reqBuf;

template <typename dtype, integer shape1, integer shape2>
void Function_mpi(std::vector<extArrType<dtype, shape1, shape2>*> phiP) {
    extArrType<dtype, shape1, shape2>& phi = *(phiP[1]);
    const Mesh& mesh = *meshp;
    //MPI_Barrier(MPI_COMM_WORLD);

    extArrType<dtype, shape1, shape2>* phiBuf;
    if (mesh.nRemotePatches > 0) {
        phiBuf = new extArrType<dtype, shape1, shape2>(mesh.nCells-mesh.nLocalCells, true);
        mpi_reqBuf[phiP[1]] = (void *) phiBuf;
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
                phiBuf->extract(bufStartFace, &phi(0), &mesh.owner(startFace), nFaces);
                MPI_Request *req = mpi_req;
                integer tag = mpi_reqField*100 + mesh.tags.at(patchID);
                //cout << patchID << " " << tag << endl;
                MPI_Isend(&(*phiBuf)(bufStartFace), size, mpi_type<dtype>(), dest, tag, MPI_COMM_WORLD, &req[mpi_reqIndex]);
                MPI_Irecv(&phi(cellStartFace), size, mpi_type<dtype>(), dest, tag, MPI_COMM_WORLD, &req[mpi_reqIndex+1]);
                mpi_reqIndex += 2;
        }
    }
    mpi_reqField = (mpi_reqField + 1) % 100;
}
template <typename dtype, integer shape1, integer shape2>
void Function_mpi_grad(std::vector<extArrType<dtype, shape1, shape2>*> phiP) {
    extArrType<dtype, shape1, shape2>& phi = *(phiP[1]);
    const Mesh& mesh = *meshp;
    //MPI_Barrier(MPI_COMM_WORLD);

    extArrType<dtype, shape1, shape2>* phiBuf;
    if (mesh.nRemotePatches > 0) {
        phiBuf = new extArrType<dtype, shape1, shape2>(mesh.nCells-mesh.nLocalCells, true);
        mpi_reqBuf[phiP[1]] = (void *) phiBuf;
    }
    for (auto& patch: mesh.boundary) {
        string patchType = patch.second.at("type");
        if (patchType == "processor" || patchType == "processorCyclic") {
            string patchID = patch.first;
            const map<string, string>& patchInfo = mesh.boundary.at(patchID);
            integer startFace, nFaces;
            tie(startFace, nFaces) = mesh.boundaryFaces.at(patch.first);
            integer cellStartFace = mesh.nInternalCells + startFace - mesh.nInternalFaces;

            integer bufStartFace = cellStartFace - mesh.nLocalCells;
            integer size = nFaces*shape1*shape2;
            integer dest = stoi(patchInfo.at("neighbProcNo"));
            
            MPI_Request *req = (MPI_Request*) mpi_req;
            integer tag = mpi_reqField*10000 + mesh.tags.at(patchID);
            //cout << "send " << patchID << " " << phi(cellStartFace) << " " << shape1 << shape2 << endl;
            MPI_Isend(&phi(cellStartFace), size, mpi_type<dtype>(), dest, tag, MPI_COMM_WORLD, &req[mpi_reqIndex]);
            MPI_Irecv(&(*phiBuf)(bufStartFace), size, mpi_type<dtype>(), dest, tag, MPI_COMM_WORLD, &req[mpi_reqIndex+1]);
            mpi_reqIndex += 2;
        }
    }
    mpi_reqField = (mpi_reqField + 1) % 100;
}

#define MPI_SPECIALIZE(func) \
template void func<>(std::vector<extArrType<scalar, 1, 1>*> phiP); \
template void func<>(std::vector<extArrType<scalar, 1, 3>*> phiP); \
template void func<>(std::vector<extArrType<scalar, 3, 1>*> phiP); \
template void func<>(std::vector<extArrType<scalar, 3, 3>*> phiP);

MPI_SPECIALIZE(Function_mpi)
MPI_SPECIALIZE(Function_mpi_grad)

void Function_mpi_allreduce(std::vector<ext_vec*> vals) {
    integer n = vals.size()/2;
    ext_vec in(n, true);
    ext_vec out(n, true);
    for (integer i = 0; i < n; i++) {
        in.copy(i, &(*vals[i])(0), 1);
    }
    #ifdef MPI_GPU
    MPI_Allreduce(&in(0), &out(0), n, mpi_type<decltype(vals[0]->type)>(), MPI_SUM, MPI_COMM_WORLD);
    for (integer i = 0; i < n; i++) {
        (*vals[i+n]).copy(0, &out(i), 1);
    }
    #else
    for (integer i = 0; i < n; i++) {
        (*vals[i+n]).copy(0, &in(i), 1);
    }
    #endif
}

void Function_mpi_allreduce_grad(std::vector<ext_vec*> vals) {
    integer n = vals.size()/3;
    for (integer i = 0; i < n; i++) {
        (*vals[i+2*n]).copy(0, &(*vals[i+n])(0), 1);
    }
}

void Function_mpi_init1() {
    const Mesh& mesh = *meshp;
    mpi_reqIndex = 0;
    mpi_reqBuf.clear();
    if (mesh.nRemotePatches > 0) {
        //MPI_Barrier(MPI_COMM_WORLD);
        mpi_req = new MPI_Request[2*3*mesh.nRemotePatches];
    }
}

void Function_mpi_init1_grad() {
    const Mesh& mesh = *meshp;
    if (mesh.nRemotePatches > 0) {
        //MPI_Barrier(MPI_COMM_WORLD);
        for (auto& kv: mpi_reqBuf) {
            ((ext_vec*)kv.second)->destroy();
            delete (ext_vec*)kv.second;
        }
    }
}

template <typename dtype, integer shape1, integer shape2>
void Function_mpi_init2(std::vector<extArrType<dtype, shape1, shape2>*> phiP) {};

template <typename dtype, integer shape1, integer shape2>
void Function_mpi_init2_grad(std::vector<extArrType<dtype, shape1, shape2>*> phiP) {
    extArrType<dtype, shape1, shape2>& phi = *(phiP[1]);
    const Mesh& mesh = *meshp;
    extArrType<dtype, shape1, shape2>* phiBuf = (extArrType<dtype, shape1, shape2> *)mpi_reqBuf[phiP[1]];
    //MPI_Barrier(MPI_COMM_WORLD);
    for (auto& patch: mesh.boundary) {
        string patchType = patch.second.at("type");
        string patchID = patch.first;
        integer startFace, nFaces;
        tie(startFace, nFaces) = mesh.boundaryFaces.at(patch.first);
        integer cellStartFace = mesh.nInternalCells + startFace - mesh.nInternalFaces;
        if (patchType == "processor" || patchType == "processorCyclic") {
            //cout << "hello " << patchID << endl;
            integer bufStartFace = cellStartFace - mesh.nLocalCells;
            //cout << "recv " << patchID << " " << phiBuf[bufStartFace*shape1*shape2] << " " << shape1 << shape2 << endl;
            phi.extract(&mesh.owner(startFace), &(*phiBuf)(bufStartFace), nFaces);
        }
    }
};

MPI_SPECIALIZE(Function_mpi_init2)
MPI_SPECIALIZE(Function_mpi_init2_grad)

void Function_mpi_init3() {}; 
void Function_mpi_init3_grad() {
    const Mesh& mesh = *meshp;
    if (mesh.nRemotePatches > 0) {
        MPI_Waitall(mpi_reqIndex, (mpi_req), MPI_STATUSES_IGNORE);
        delete[] mpi_req;
        //MPI_Barrier(MPI_COMM_WORLD);
    }
}

void Function_mpi_end() {
    Function_mpi_init3_grad();
    Function_mpi_init1_grad();
}

