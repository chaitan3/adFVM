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

void Operator::grad(const scalar* phi, arr& gradPhi, integer index) {
    const Mesh& mesh = this->mesh;
    
    integer p = mesh.owner(index);
    integer n = mesh.neighbour(index);
    scalar wp = mesh.areas(index)/mesh.volumes(p);
    scalar wn = mesh.areas(index)/mesh.volumes(n);
    for (integer i = 0; i < gradPhi.shape[2]; i++) {
        for (integer j = 0; j < 3; j++) {
            gradPhi(p, i, j) += phi[i]*mesh.normals(index, j)*wp;
            gradPhi(n, i, j) -= phi[i]*mesh.normals(index, j)*wn;
        }
    }
}

void Operator::div(const scalar* phi, arr& divPhi, integer index) {
    const Mesh& mesh = this->mesh;
    
    integer p = mesh.owner(index);
    integer n = mesh.neighbour(index);
    scalar wp = mesh.areas(index)/mesh.volumes(p);
    scalar wn = mesh.areas(index)/mesh.volumes(n);
    for (integer i = 0; i < divPhi.shape[1]; i++) {
        divPhi(p, i) += phi[i]*wp;
        divPhi(n, i) -= phi[i]*wn;
    }
}

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

