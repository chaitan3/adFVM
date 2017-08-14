#define NO_IMPORT_ARRAY
#include "parallel.hpp"
#include "mesh.hpp"

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
                phiBuf->extract(bufStartFace, &mesh.owner(startFace), &phi(0), nFaces);
                integer tag = mpi_reqField*100 + mesh.tags.at(patchID);
                //cout << patchID << " " << tag << endl;
                MPI_Isend(&(*phiBuf)(bufStartFace), size, mpi_type<dtype>(), dest, tag, MPI_COMM_WORLD, &mpi_req[mpi_reqIndex]);
                MPI_Irecv(&phi(cellStartFace), size, mpi_type<dtype>(), dest, tag, MPI_COMM_WORLD, &mpi_req[mpi_reqIndex+1]);
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
            
            integer tag = mpi_reqField*10000 + mesh.tags.at(patchID);
            //cout << "send " << patchID << " " << phi(cellStartFace) << " " << shape1 << shape2 << endl;
            MPI_Isend(&phi(cellStartFace), size, mpi_type<dtype>(), dest, tag, MPI_COMM_WORLD, &mpi_req[mpi_reqIndex]);
            MPI_Irecv(&(*phiBuf)(bufStartFace), size, mpi_type<dtype>(), dest, tag, MPI_COMM_WORLD, &mpi_req[mpi_reqIndex+1]);
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
