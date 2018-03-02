#define NO_IMPORT_ARRAY
#include "matop_cuda.hpp"
#define nrhs 5


Matop::Matop() {
    const Mesh& mesh = *meshp;
    assert (mesh.nProcs == 1);
    gpuErrorCheck(cusolverSpCreate(&handle));
    gpuErrorCheck(cusparseCreate(&cusparseHandle));

    int n = mesh.nInternalCells;
    arrType<int, 1> indptr(n+1);
    arrType<int, 1> indices(n*7);
    int index = 0;
    indptr(index) = 0;
    for (int i = 0; i < n; i++) {
        indices(index) = i;
        index += 1;
        for (PetscInt k = 0; k < 6; k++) {
            int col = mesh.cellNeighbours(i, k);
            if (col > -1) {
                indices(index) = col;
                index += 1;
            } 
        }
        indptr(i+1) = index;
    }
    this->nnz = index;
    this->indptr = extArrType<int, 1>(n, indptr.data);
    this->indices = extArrType<int, 1>(n, indices.data);
    this->volumes = ext_vec(n, mesh.volumes.data);
    this->cellFaces = extArrType<scalar, 6>(n, mesh.cellFaces.data);
    this->data = ext_vec(n);
}

Matop::~Matop () {
    gpuErrorCheck(cusolverSpDestroy(handle));
    gpuErrorCheck(cusparseDestroy(cusparseHandle));
}

__global__ void computeFaceData(const integer n, const scalar* __restrict__ faceData, const scalar* __restrict__ volumes, const scalar* __restrict__ dt, const integer* __restrict__ cellFaces, const int* __restrict__ indptr, scalar* __restrict__ data) {
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    for (; i < n; i += blockDim.x*gridDim.x) {
        scalar cellData = 0.;
        for (int k = 0; k < 6; k++) {
            integer f = cellFaces(i, k);
            int start = indptr[i];
            int end = indptr[i+1];
            if (k < (end-start)) {
                scalar neighbourData = -faceData[f]/volumes(i)*dt[0];
                data[start+k+1] = neighbourData;
                cellData += neighbourData;
            }
            data[start] = cellData;
        }
    }
}

int Matop::heat_equation(vector<ext_vec*> u, const ext_vec& DTF, const ext_vec& dt_vec, vector<ext_vec*> un) {
    long long start = current_timestamp();
    const Mesh& mesh = *meshp;

    cusparseMatDescr_t A = NULL;
    gpuErrorCheck(cusparseCreateMatDescr(&A));
    scalar tol = 1e-12;
    int reorder = 2; 
    int singularity = 0; 

    int n = mesh.nInternalCells;
    computeFaceData<<<GPU_BLOCKS_PER_GRID, GPU_THREADS_PER_BLOCK>>>(n, &DTF(0), &volumes(0), &dt(0), &cellFaces(0), &indptr(0), &data(0));
    for (integer i = 0; i < nrhs; i++) {
        gpuErrorCheck(cusolverSpDcsrlsvlu(handle, n, nnz, A, data, indptr, indices, u[i]->data, tol, reorder, un[i]->data, &singularity));
    }
    return 0;
}



