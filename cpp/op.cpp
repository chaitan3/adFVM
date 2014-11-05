#include "op.hpp"

arr Operator::internal_sum(const arr& phi) {
    return ROWDIV(
            (ROWMUL(phi, mesh.areas).matrix() * mesh.sumOp.transpose()).array(),
            mesh.volumes
            );
}

arr Operator::grad(const arr& phi) {
    // if phi is 1D
    //arr gradF = ROWMUL(mesh.normals, phi);
    // if phi is 3D
    arr gradF = outerProduct(mesh.normals, phi); 
    return internal_sum(gradF);
}

arr Operator::div(const arr& phi) {
    return internal_sum(phi);
}

arr Operator::snGrad(const arr& phi) {
    arr phiN = slice(phi, mesh.neighbour);
    arr phiP = slice(phi, mesh.owner);
    return ROWMUL(phiN-phiP, mesh.deltas);
}

arr Operator::laplacian(const arr& phi, const arr& DT) {
    return internal_sum(DT * snGrad(phi));
}

