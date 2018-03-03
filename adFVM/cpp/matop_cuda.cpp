#define NO_IMPORT_ARRAY
#include "matop_cuda.hpp"
#define nrhs 5


//Matop::Matop() {
//    const Mesh& mesh = *meshp;
//    assert (mesh.nProcs == 1);
//    auto status1 = cusolverSpCreate(&handle);
//    assert(status1 == CUSOLVER_STATUS_SUCCESS);
//    auto status2 = cusparseCreate(&cusparseHandle);
//    assert(status2 == CUSPARSE_STATUS_SUCCESS);
//
//    int n = mesh.nInternalCells;
//    arrType<int, 1> indptr(n+1);
//    arrType<int, 1> indices(n*7);
//    int index = 0;
//    indptr(0) = 0;
//    for (int i = 0; i < n; i++) {
//        indices(index) = i;
//        index += 1;
//        for (int k = 0; k < 6; k++) {
//            int col = mesh.cellNeighbours(i, k);
//            if (col > -1) {
//                indices(index+k) = col;
//                index += 1;
//            } 
//        }
//        indptr(i+1) = index;
//    }
//    this->nnz = index;
//    this->indptr = extArrType<int, 1>(n+1, indptr.data);
//    this->indices = extArrType<int, 1>(nnz, indices.data);
//    this->volumes = ext_vec(n, mesh.volumes.data);
//    this->cellFaces = extArrType<integer, 6>(n, mesh.cellFaces.data);
//    this->cellNeighbours = extArrType<integer, 6>(n, mesh.cellNeighbours.data);
//    this->data = ext_vec(nnz);
//}
//
//Matop::~Matop () {
//    auto status1 = cusolverSpDestroy(handle);
//    assert(status1 == CUSOLVER_STATUS_SUCCESS);
//    auto status2 = cusparseDestroy(cusparseHandle);
//    assert(status1 == CUSPARSE_STATUS_SUCCESS);
//}
//
//__global__ void computeFaceData(const integer n, const scalar* __restrict__ faceData, const scalar* __restrict__ volumes, const scalar* __restrict__ dt, const integer* __restrict__ cellFaces, const integer* __restrict__ cellNeighbours, const int* __restrict__ indptr, scalar* __restrict__ data) {
//    int i = threadIdx.x + blockDim.x*blockIdx.x;
//    for (; i < n; i += blockDim.x*gridDim.x) {
//        scalar cellData = 0.;
//        int start = indptr[i];
//        int curr = start + 1;
//        for (int k = 0; k < 6; k++) {
//            if (cellNeighbours[i*6+k] > -1) {
//                integer f = cellFaces[i*6 + k];
//                scalar neighbourData = -faceData[f]/volumes[i]*dt[0];
//                data[curr] = neighbourData;
//                curr++;
//                cellData += neighbourData;
//            }
//        }
//        data[start] = cellData;
//    }
//}
//
//int Matop::heat_equation(vector<ext_vec*> u, const ext_vec& DTF, const ext_vec& dt_vec, vector<ext_vec*> un) {
//    long long start = current_timestamp();
//    const Mesh& mesh = *meshp;
//
//    cusparseMatDescr_t A = NULL;
//    auto status1 = cusparseCreateMatDescr(&A);
//    assert(status1 == CUSPARSE_STATUS_SUCCESS);
//    scalar tol = 1e-12;
//    int reorder = 0; 
//    int singularity = 0; 
//
//    int n = mesh.nInternalCells;
//    computeFaceData<<<GPU_BLOCKS_PER_GRID, GPU_THREADS_PER_BLOCK>>>(n, &DTF(0), &volumes(0), &dt_vec(0), &cellFaces(0), &cellNeighbours(0), &indptr(0), &data(0));
//    for (integer i = 0; i < nrhs; i++) {
//        //#ifdef GPU_DOUBLE
//        //    auto status2 = cusolverSpDcsrlsvqr(handle, n, nnz, A, &data(0), &indptr(0), &indices(0), u[i]->data, tol, reorder, un[i]->data, &singularity);
//        //#else
//        //    auto status2 = cusolverSpScsrlsvqr(handle, n, nnz, A, &data(0), &indptr(0), &indices(0), u[i]->data, tol, reorder, un[i]->data, &singularity);
//        //#endif
//        //cout << "status: " << status2 << endl;
//
//        //assert(status2 == CUSOLVER_STATUS_SUCCESS);
//    }
//    return 0;
//}


Matop::Matop() {
    const Mesh& mesh = *meshp;
    assert (mesh.nProcs == 1);

    int n = mesh.nInternalCells;
    this->volumes = ext_vec(n, mesh.volumes.data);
    this->cellFaces = extArrType<integer, 6>(n, mesh.cellFaces.data);
    this->cellNeighbours = extArrType<integer, 6>(n, mesh.cellNeighbours.data);
}

Matop::~Matop () {
}

__global__ void jacobiIteration(const integer nInternalCells, const scalar* phi, const integer* cellNeighbours, const scalar* neighbourData, const scalar* cellData, scalar* phiN) {
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    for (; i < nInternalCells; i += blockDim.x*gridDim.x) {
        scalar lapPhi = 0.;
        for (int j = 0; j < 6; j++) {
            integer k = i*6+j;
            integer neighbour = cellNeighbours[k];
            if (neighbour > -1) {
                lapPhi += neighbourData[k]*phi[neighbour];
            }
        }
        phiN[i] = (phi[i]-lapPhi)/(cellData[i] + 1.);
    }
}

__global__ void residual(const integer nInternalCells, const scalar* phi, const integer* cellNeighbours, const scalar* neighbourData, const scalar* cellData, scalar* res) {
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    for (; i < nInternalCells; i += blockDim.x*gridDim.x) {
        scalar lapPhi = 0.;
        for (int j = 0; j < 6; j++) {
            integer k = i*6+j;
            integer neighbour = cellNeighbours[k];
            if (neighbour > -1) {
                lapPhi += neighbourData[k]*phi[neighbour];
            }
        }
        scalar tmp = (phi[i]-lapPhi) - (cellData[i] + 1.)*phi[i];
        atomicAdd(res, tmp*tmp);
    }
}

__global__ void computeCellData(const integer nInternalCells, const scalar* dt_p, const integer* cellFaces, const integer* cellNeighbours, const scalar* volumes, const scalar* faceData, scalar* neighbourData, scalar* cellData) {
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    scalar dt = dt_p[0];
    for (; i < nInternalCells; i += blockDim.x*gridDim.x) {
        scalar data = 0.;
        scalar V = volumes[i];
        for (int j = 0; j < 6; j++) {
            integer k = i*6+j;
            integer neighbour = cellNeighbours[k];
            if (neighbour > -1) {
                integer p = cellFaces[k];
                scalar tmp = -faceData[p]*dt/V;
                neighbourData[k] = tmp;
                data -= tmp;
            }
        }
        cellData[i] = data;
    }
}

int Matop::heat_equation(vector<ext_vec*> u, const ext_vec& DTF, const ext_vec& dt_vec, vector<ext_vec*> un) {
    long long start = current_timestamp();
    const Mesh& mesh = *meshp;

    int n = mesh.nInternalCells;
    extArrType<scalar, 6> neighbourData(n, false, true, 0);
    extArrType<scalar, 1> cellData(n, false, true, 0);
    computeCellData<<<GPU_BLOCKS_PER_GRID, GPU_THREADS_PER_BLOCK>>>(n, &dt_vec(0), &cellFaces(0), &cellNeighbours(0), &volumes(0), &DTF(0), &neighbourData(0), &cellData(0));
    scalar* res_g;
    cudaMalloc(&res_g, sizeof(scalar)*1);
    extArrType<scalar, 1> tmp(n, false, true, 0);
    for (int i = 0; i < nrhs; i++) {
        scalar res0, resf;
        cudaMemcpy(un[i]->data, u[i]->data, n*sizeof(scalar), cudaMemcpyDeviceToDevice);
        for (int j = 0; j < 1000; j++) {
            scalar* buff,*buff2;
            if (j % 2 == 0) {
                buff = un[i]->data;
                buff2 = &tmp(0);
            } else {
                buff = &tmp(0);
                buff2 = un[i]->data;
            }
            scalar res = 0.;
            cudaMemcpy(res_g, &res, sizeof(scalar)*1, cudaMemcpyHostToDevice);
            residual<<<GPU_BLOCKS_PER_GRID, GPU_THREADS_PER_BLOCK>>>(n, buff, &cellNeighbours(0), &neighbourData(0), &cellData(0), res_g);
            cudaMemcpy(&res, res_g, sizeof(scalar)*1, cudaMemcpyDeviceToHost);
            //printf("res %d %d: %f\n", j, i, sqrt(res/n));
            if (j == 0) res0 = res;
            if (j == 999) resf = res;
            jacobiIteration<<<GPU_BLOCKS_PER_GRID, GPU_THREADS_PER_BLOCK>>>(n, buff, &cellNeighbours(0), &neighbourData(0), &cellData(0), buff2);
        }
        printf("res ratio %d: %f\n", i, sqrt(res0/resf));
    }
    cudaFree(res_g);

    return 0;
}



