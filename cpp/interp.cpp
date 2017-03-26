#include "interp.hpp"

template<integer shape1, integer shape2>
void Interpolator::central(const arrType<scalar, shape1, shape2>& phi, scalar* phiF, integer index) {
    const Mesh& mesh = *this->mesh;
    integer p = mesh.owner(index);
    integer n = mesh.neighbour(index);
    scalar w = mesh.weights(index);
    //integer w = mesh.weights(index);
    for (integer i = 0; i < shape1; i++) {
        for (integer j = 0; j < shape2; j++) {
            phiF[i*phi.shape[2]+j] = phi(n, i, j)*(1-w) + phi(p, i, j)*w;
        }
    }
}

template<integer shape1, integer shape2>
void Interpolator::average(const arrType<scalar, shape1, shape2>& phi, scalar* phiF, integer index) {
    const Mesh& mesh = *this->mesh;
    integer p = mesh.owner(index);
    integer n = mesh.neighbour(index);
    scalar w = 0.5;
    //integer w = mesh.weights(index);
    for (integer i = 0; i < shape1; i++) {
        for (integer j = 0; j < shape2; j++) {
            phiF[i*phi.shape[2]+j] = phi(n, i, j)*(1-w) + phi(p, i, j)*w;
        }
    }
}

template<integer shape1>
void Interpolator::firstOrder(const arrType<scalar, shape1>& phi, const arrType<scalar, shape1, 3>& gradPhi, scalar *phiF, integer index, integer swap) {
    const Mesh& mesh = *this->mesh;
    integer p;
    if (swap) {
        p = mesh.neighbour(index);
    } else {
        p = mesh.owner(index);
    }
    for (integer i = 0; i < shape1; i++) {
        phiF[i] = phi(p, i);
    }
}

template<integer shape1>
void Interpolator::secondOrder(const arrType<scalar, shape1>& phi, const arrType<scalar, shape1, 3>& gradPhi, scalar *phiF, integer index, integer swap) {
    const Mesh& mesh = *this->mesh;
    integer p, n;
    if (swap) {
        n = mesh.owner(index);
        p = mesh.neighbour(index);
    } else {
        p = mesh.owner(index);
        n = mesh.neighbour(index);
    }
    for (integer i = 0; i < shape1; i++) {
        scalar phiC = phi(p, i);
        scalar phiD = phi(n, i);
        phiF[i] = phiC + (phiD-phiC)*mesh.linearWeights(swap, index);
        for (integer j = 0; j < 3; j++) {
            phiF[i] += mesh.quadraticWeights(swap, index, j)*gradPhi(p, i, j);
        }
    }
}
