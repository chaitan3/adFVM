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
            phiF[i*shape2+j] = phi(n, i, j)*(1-w) + phi(p, i, j)*w;
        }
    }
}
template void Interpolator::central(const arrType<scalar, 1, 1>& phi, scalar* phiF, integer index);
template void Interpolator::central(const arrType<scalar, 3, 1>& phi, scalar* phiF, integer index);

template<integer shape1, integer shape2>
void Interpolator::average(const arrType<scalar, shape1, shape2>& phi, scalar* phiF, integer index) {
    const Mesh& mesh = *this->mesh;
    integer p = mesh.owner(index);
    integer n = mesh.neighbour(index);
    scalar w = 0.5;
    //integer w = mesh.weights(index);
    for (integer i = 0; i < shape1; i++) {
        for (integer j = 0; j < shape2; j++) {
            phiF[i*shape2+j] = phi(n, i, j)*(1-w) + phi(p, i, j)*w;
        }
    }
}
template void Interpolator::average(const arrType<scalar, 1, 3>& phi, scalar* phiF, integer index);
template void Interpolator::average(const arrType<scalar, 3, 3>& phi, scalar* phiF, integer index);

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
template void Interpolator::firstOrder(const arrType<scalar, 1>& phi, const arrType<scalar, 1, 3>& gradPhi, scalar *phiF, integer index, integer swap);
template void Interpolator::firstOrder(const arrType<scalar, 3>& phi, const arrType<scalar, 3, 3>& gradPhi, scalar *phiF, integer index, integer swap);

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
template void Interpolator::secondOrder(const arrType<scalar, 1>& phi, const arrType<scalar, 1, 3>& gradPhi, scalar *phiF, integer index, integer swap);
template void Interpolator::secondOrder(const arrType<scalar, 3>& phi, const arrType<scalar, 3, 3>& gradPhi, scalar *phiF, integer index, integer swap);
