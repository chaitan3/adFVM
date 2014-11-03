#include "interp.hpp"

arr Interpolator::central(const arr& phi) {
    arr phiN = slice(phi, mesh.neighbour);
    arr phiP = slice(phi, mesh.owner);
    phiN.rowwise() *= (1-mesh.weights).row(0);
    phiP.rowwise() *= mesh.weights.row(0);
    return phiN + phiP;
}
arr Interpolator::TVD(const arr& phi) {
    return this->central(phi);
}
