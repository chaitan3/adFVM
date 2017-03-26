#include "op.hpp"

// other implementation
/*inline arr Operator::internal_sum(const arr& phi) {*/
    //arr phiC = arr::Zero(phi.rows(), mesh.nInternalCells);
    //for (int j = 0; j < mesh.nFaces; j++) {
        //for (int i = 0; i < phi.rows(); i++) {
            //int p = mesh.owner(0, j);
            //phiC(i, p) += phi(i, j)*mesh.areas(0, j)/mesh.volumes(0, p);
            //if (j < mesh.nInternalFaces) {
                //int n = mesh.neighbour(0, j);
                //phiC(i, n) -= phi(i, j)*mesh.areas(0, j)/mesh.volumes(0, n);
            //}
        //}
    //}
   //return phiC;
//}
//

template<integer shape1>
void Operator::grad(const scalar* phi, arrType<scalar, shape1, 3>& gradPhi, integer index, bool neighbour) {
    const Mesh& mesh = *this->mesh;
    
    integer p = mesh.owner(index);
    integer n = mesh.neighbour(index);
    scalar wp = mesh.areas(index)/mesh.volumes(p);
    scalar wn;
    if (neighbour) {
        wn = mesh.areas(index)/mesh.volumes(n);
    }
    //cout << index << " " << mesh.nInternalFaces << " " << mesh.nFaces << " " << mesh.nInternalCells << endl;
    //cout << p << " " << mesh.areas(index) << " " << mesh.areas.data[index] << endl;
    for (integer i = 0; i < shape1; i++) {
        for (integer j = 0; j < 3; j++) {
            gradPhi(p, i, j) += phi[i]*mesh.normals(index, j)*wp;
            if (neighbour) {
                gradPhi(n, i, j) -= phi[i]*mesh.normals(index, j)*wn;
            }
        }
    }
}
template void Operator::grad(const scalar* phi, arrType<scalar, 1, 3>& gradPhi, integer index, bool neighbour);
template void Operator::grad(const scalar* phi, arrType<scalar, 3, 3>& gradPhi, integer index, bool neighbour);

template<integer shape1>
void Operator::div(const scalar* phi, arrType<scalar, shape1>& divPhi, integer index, bool neighbour) {
    const Mesh& mesh = *this->mesh;
    
    integer p = mesh.owner(index);
    integer n = mesh.neighbour(index);
    scalar wp = mesh.areas(index)/mesh.volumes(p);
    scalar wn;
    if (neighbour) {
        wn = mesh.areas(index)/mesh.volumes(n);
    }
    for (integer i = 0; i < shape1; i++) {
        divPhi(p, i) += phi[i]*wp;
        if (neighbour) {
            divPhi(n, i) -= phi[i]*wn;
        }
    }
}
template void Operator::div(const scalar* phi, arrType<scalar, 1>& divPhi, integer index, bool neighbour);
template void Operator::div(const scalar* phi, arrType<scalar, 3>& divPhi, integer index, bool neighbour);

template<integer shape1>
void Operator::absDiv(const scalar* phi, arrType<scalar, shape1>& divPhi, integer index, bool neighbour) {
    const Mesh& mesh = *this->mesh;
    
    integer p = mesh.owner(index);
    integer n = mesh.neighbour(index);
    scalar wp = mesh.areas(index)/mesh.volumes(p);
    scalar wn;
    if (neighbour) {
        wn = mesh.areas(index)/mesh.volumes(n);
    }
    for (integer i = 0; i < shape1; i++) {
        divPhi(p, i) += phi[i]*wp;
        if (neighbour) {
            divPhi(n, i) += phi[i]*wn;
        }
    }
}
template void Operator::absDiv(const scalar* phi, arrType<scalar, 1>& divPhi, integer index, bool neighbour);

template<integer shape1>
void Operator::snGrad(const arrType<scalar, shape1>& phi, scalar* snGradPhi, integer index) {
    const Mesh& mesh = *this->mesh;
    
    integer p = mesh.owner(index);
    integer n = mesh.neighbour(index);
    integer d = mesh.deltas(index);
    for (integer i = 0; i < shape1; i++) {
        snGradPhi[i] = (phi(n, i)-phi(p, i))*d;
    }
}
template void Operator::snGrad(const arrType<scalar, 1>& phi, scalar* snGradPhi, integer index);
template void Operator::snGrad(const arrType<scalar, 3>& phi, scalar* snGradPhi, integer index);

//arr Operator::div(const arr& phi) {
    //return internal_sum(phi);
//}

//arr Operator::snGrad(const arr& phi) {
    //arr phiN = slice(phi, mesh.neighbour);
    //arr phiP = slice(phi, mesh.owner);
    //return ROWMUL(phiN-phiP, mesh.deltas);
//}

//arr Operator::laplacian(const arr& phi, const arr& DT) {
    //return internal_sum(DT * snGrad(phi));
//}

