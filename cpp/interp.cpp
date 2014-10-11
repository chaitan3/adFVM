#include "interp.hpp"

arr Interpolator::central(const arr& phi) {
    arr phiF(phi.rows(), mesh.nFaces);
    arr phiN = slice(phi, mesh.neighbour);
    arr phiP = slice(phi, mesh.owner);
    for (int i = 0; i < phiF.rows(); i++)
        phiF.row(i) = phiN.row(i) + phiP.row(i);
        /phiF.rowwise() = phiN.rowwise() + phiP.rowwise();
    //phiF.rowwise() = (mesh.weights.row(0))*phiN.rowwise() + mesh.weights.row(0)*phiP.rowwise();
    return phiF;
}
arr Interpolator::TVD(const arr& phi) {
    arr phiF(phi.rows(), mesh.nFaces);
    return this->central(phi);
}


   
