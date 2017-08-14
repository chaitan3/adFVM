#include <iostream>
#include "mpi.h"

using namespace std;

__global__ void set_value(double* a) {
    a[0] = 4;
}

int main(int argc, char** argv) {
    int rank;
    cudaSetDevice(0);
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cout << rank << endl;
    double* a;
    double* b;
    cudaMalloc(&a, sizeof(double));
    set_value<<<1, 1>>>(a);
    cudaMalloc(&b, sizeof(double));
    //cudaMemset(a, rank + 4, sizeof(double));
    if (rank == 0) {
        cout << "here0" << endl;
        MPI_Send(&a, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
    } else {
        cout << "here1" << endl;
        MPI_Recv(&b, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, NULL);
        double c[1];
        cudaMemcpy(c, b, sizeof(double), cudaMemcpyDeviceToHost);
        cout << "Recv: " << c[0] << endl;
    }
    cudaFree(a);
    cudaFree(b);
    MPI_Finalize();
}
