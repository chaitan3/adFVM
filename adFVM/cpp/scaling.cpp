#define NO_IMPORT_ARRAY
#include "scaling.hpp"
#ifdef MATOP
    #include "matop.hpp"
#endif

extern "C"{
void dsyev_( char* jobz, char* uplo, int* n, double* a, int* lda,
    double* w, double* work, int* lwork, int* info );
void ssyev_( char* jobz, char* uplo, int* n, float* a, int* lda,
    float* w, float* work, int* lwork, int* info );
}

template<typename dtype> void eigenvalue_solver (char* jobz, char* uplo, int* n, dtype* a, int* lda,
dtype* w, dtype* work, int* lwork, int* info );

template<> void eigenvalue_solver (char* jobz, char* uplo, int* n, float* a, int* lda,
float* w, float* work, int* lwork, int* info ) {
    ssyev_(jobz, uplo, n, a, lda, w, work, lwork, info);
}
template<> void eigenvalue_solver (char* jobz, char* uplo, int* n, double* a, int* lda,
double* w, double* work, int* lwork, int* info ) {
    dsyev_(jobz, uplo, n, a, lda, w, work, lwork, info);
}

#ifdef GPU
void Function_get_max_eigenvalue(vector<gpuArrType<scalar, 5, 5>*> phiP) {
    gpuArrType<scalar, 5, 5>& phi = *phiP[0];
    gvec& eigPhi = *((gvec*) phiP[1]);
    syevjInfo_t params;
    int n = phi.shape;
    int* info;
    int lwork = 0;
    int size = sizeof(scalar);
    scalar* W, *work;
    gpuErrorCheck(cudaMalloc(&W, size*5*n));
    gpuErrorCheck(cudaMalloc(&info, sizeof(int)*n));

    cusolverStatus_t status;
    status = cusolverDnCreateSyevjInfo(&params);
    assert(status == CUSOLVER_STATUS_SUCCESS);
    status = cusolverDnSsyevjBatched_bufferSize(cusolver_handle, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_UPPER, 5, phi.data, 5, W, &lwork, params, n);
    gpuErrorCheck(cudaMalloc(&work, size*lwork));
    status = cusolverDnSsyevjBatched(cusolver_handle, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_UPPER, 5, phi.data, 5, W, work, lwork, info, params, n);
    gpuErrorCheck(cudaDeviceSynchronize());
    assert(status == CUSOLVER_STATUS_SUCCESS);

    gpuErrorCheck(cudaMemcpy2D(eigPhi.data, size, W + 4, 5*size, size, n, cudaMemcpyDeviceToDevice));
    gpuErrorCheck(cudaFree(work));
    gpuErrorCheck(cudaFree(W));
    gpuErrorCheck(cudaFree(info));
}
#else
void Function_get_max_eigenvalue(vector<extArrType<scalar, 5, 5>*> phiP) {
    extArrType<scalar, 5, 5>& phi = *phiP[0];
    ext_vec& eigPhi = *((ext_vec*) phiP[1]);
    char jobz = 'N';
    char uplo = 'U';
    int n = 5;
    int lda = 5;
    int lwork = 3*n-1;
    scalar work[lwork];
    int info;
    scalar w[5];

    #ifdef GPU
        arrType<scalar, 5, 5> phiWork(phi.shape, phi.toHost());
        phiWork.ownData = true;
        arrType<scalar, 1> eigPhiWork(phi.shape);
    #else
        arrType<scalar, 5, 5>& phiWork = phi;
        arrType<scalar, 1>& eigPhiWork = eigPhi;
    #endif

    for (int i = 0; i < phi.shape; i++) {
        eigenvalue_solver<scalar>(&jobz, &uplo, &n, &phiWork(i), &lda, w, work, &lwork, &info);
        assert(info == 0);
        eigPhiWork(i) = w[4];
    }
    eigPhi.toDevice(eigPhiWork.data);
}
#endif

void Function_apply_adjoint_viscosity(vector<ext_vec*> phiP) {
    // inputs
    vector<ext_vec*> u, un;
    for (int i = 0; i < 5; i++) {
        u.push_back(phiP[i]);
        un.push_back(phiP[i+7]);
    }
    ext_vec& DT = *phiP[5];
    ext_vec& dt_vec = *phiP[6];

    #ifdef MATOP
        #ifdef GPU
            scalar* dt_vec_data = dt_vec.toHost();
            scalar dt = dt_vec_data[0];
            delete[] dt_vec_data;
            vec DTWork(DT.shape, DT.toHost());
            DTWork.ownData = true;
            vector<vec*> uWork, unWork;
            for (int i = 0; i < 5; i++) {
                uWork.push_back(new vec(u[i]->shape, u[i]->toHost()));
                uWork[i]->ownData = true;
                unWork.push_back(new vec(un[i]->shape, un[i]->toHost()));
                unWork[i]->ownData = true;
            }
        #else
            scalar dt = dt_vec(0);
            vector<vec*> uWork = u;
            vector<vec*> unWork = un;
            vec& DTWork = DT;
        #endif
        int error = matop->heat_equation(uWork, DTWork, dt, unWork);
        if (error) {
            cout << "petsc error " << error << endl;
        }
        for (int i = 0; i < 5; i++) {
            un[i]->toDevice(unWork[i]->data);
            #ifdef GPU
                delete uWork[i];
                delete unWork[i];
            #endif
        }
    #else
        cout << "matop not available" << endl;
        for (int i = 0; i < 5; i++) {
            un[i]->copy(0, u[i]->data, u[i]->size);
        }
    #endif
}

