#include "op.hpp"

arr Operator::internal_sum(const arr& phi) {
    return ROWDIV(
            (ROWMUL(phi, mesh.areas).matrix() * mesh.sumOpT).array(),
            mesh.volumes
            );
}

arr Operator::grad(const arr& phi) {
    arr gradF;
    if (phi.rows() == 1) {
        gradF = ROWMUL(mesh.normals, phi);
    } else {
        gradF = outerProduct(mesh.normals, phi); 
    }
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

