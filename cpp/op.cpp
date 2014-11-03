#include "op.hpp"

arr Operator::internal_sum(const arr& phi) {
    return ((phi.rowwise() * mesh.areas.row(0)).matrix() \
            * mesh.sumOp.transpose()).array() \
            .rowwise()/mesh.volumes.row(0);
}

arr Operator::grad(const arr& phi) {
    // if phi is 1D
    arr gradF = mesh.normals.rowwise() * phi.row(0);
    // if phi is 3D
    // ??
    return internal_sum(gradF);
}

arr Operator::div(const arr& phi) {
    return internal_sum(phi);
}

arr Operator::snGrad(const arr& phi) {
    arr phiN = slice(phi, mesh.neighbour);
    arr phiP = slice(phi, mesh.owner);
    arr gradFdotN = (phiN-phiP).rowwise()/mesh.deltas.row(0);
    return gradFdotN;
}

arr Operator::laplacian(const arr& phi) {
    return internal_sum(snGrad(phi));
}


