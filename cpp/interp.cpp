#include "interp.hpp"

void Interpolator::central(const arr& phi, scalar* phiF, integer index) {
    const Mesh& mesh = *this->mesh;
    integer p = mesh.owner(index);
    integer n = mesh.neighbour(index);
    scalar w = mesh.weights(index);
    for (integer i = 0; i < phi.shape[1]; i++) {
        for (integer j = 0; j < phi.shape[2]; j++) {
            phiF[i*phi.shape[2]+j] = phi(n, i, j)*(1-w) + phi(p, i, j)*w;
        }
    }
}

//void Interpolator::firstOrder(const arr& phi, const arr& gradPhi, scalar *phiF, integer index, integer swap) {
/*void Interpolator::secondOrder(const arr& phi, const arr& gradPhi, scalar *phiF, integer index, integer swap) {*/
    //const Mesh& mesh = *this->mesh;
    //integer p;
    //if (swap) {
        //p = mesh.neighbour(index);
    //} else {
        //p = mesh.owner(index);
    //}
    //for (integer i = 0; i < phi.shape[1]; i++) {
        //phiF[i] = phi(p, i);
    //}
/*}*/

void Interpolator::secondOrder(const arr& phi, const arr& gradPhi, scalar *phiF, integer index, integer swap) {
    const Mesh& mesh = *this->mesh;
    integer p, n;
    if (swap) {
        n = mesh.owner(index);
        p = mesh.neighbour(index);
    } else {
        p = mesh.owner(index);
        n = mesh.neighbour(index);
    }
    for (integer i = 0; i < phi.shape[1]; i++) {
        scalar phiC = phi(p, i);
        scalar phiD = phi(n, i);
        phiF[i] = phiC + (phiD-phiC)*mesh.linearWeights(swap, index);
        for (integer j = 0; j < 3; j++) {
            phiF[i] += mesh.quadraticWeights(swap, index, j)*gradPhi(p, i, j);
        }
    }
}
