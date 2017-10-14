#define NO_IMPORT_ARRAY
#include "scaling.hpp"
#ifdef MATOP
    #include "matop.hpp"
#endif

#ifdef GPU

#ifdef GPU_DOUBLE
    #define gpu_eigenvalue_solver_buffer cusolverDnDsyevjBatched_bufferSize
    #define gpu_eigenvalue_solver cusolverDnDsyevjBatched
#else
    #define gpu_eigenvalue_solver_buffer cusolverDnSsyevjBatched_bufferSize
    #define gpu_eigenvalue_solver cusolverDnSsyevjBatched
#endif

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
    status = gpu_eigenvalue_solver_buffer(cusolver_handle, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_UPPER, 5, phi.data, 5, W, &lwork, params, n);
    gpuErrorCheck(cudaMalloc(&work, size*lwork));
    status = gpu_eigenvalue_solver(cusolver_handle, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_UPPER, 5, phi.data, 5, W, work, lwork, info, params, n);
    gpuErrorCheck(cudaDeviceSynchronize());
    assert(status == CUSOLVER_STATUS_SUCCESS);

    gpuErrorCheck(cudaMemcpy2D(eigPhi.data, size, W + 4, 5*size, size, n, cudaMemcpyDeviceToDevice));
    gpuErrorCheck(cudaFree(work));
    gpuErrorCheck(cudaFree(W));
    gpuErrorCheck(cudaFree(info));
}

#else

extern "C"{
#ifdef CPU_FLOAT32
void ssyev_( char* jobz, char* uplo, int* n, float* a, int* lda,
    float* w, float* work, int* lwork, int* info );
#define eigenvalue_solver ssyev_
#else
void dsyev_( char* jobz, char* uplo, int* n, double* a, int* lda,
    double* w, double* work, int* lwork, int* info );
#define eigenvalue_solver dsyev_
#endif
}

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

    for (int i = 0; i < phi.shape; i++) {
        eigenvalue_solver(&jobz, &uplo, &n, &phi(i), &lda, w, work, &lwork, &info);
        assert(info == 0);
        eigPhi(i) = w[4];
    }
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
        int error = matop->heat_equation(u, DT, dt_vec, un);
        if (error) {
            cout << "petsc error " << error << endl;
            exit(1);
        }
    #else
        cout << "matop not available" << endl;
        for (int i = 0; i < 5; i++) {
            un[i]->copy(0, u[i]->data, u[i]->size);
        }
    #endif
}

