#include "op.hpp"

// vectorized implementation
inline arr Operator::internal_sum(const arr& phi) {
    return ROWDIV(
            (ROWMUL(phi, mesh.areas).matrix() * mesh.sumOpT).array(),
            mesh.volumes
            );
}

// other implementation
//inline arr Operator::internal_sum(const arr& phi) {
//    arr phiC = arr::Zero(phi.rows(), mesh.nInternalCells);
//    for (int j = 0; j < mesh.nFaces; j++) {
//        for (int i = 0; i < phi.rows(); i++) {
//            int p = mesh.owner(0, j);
//            phiC(i, p) += phi(i, j)*mesh.areas(0, j)/mesh.volumes(0, p);
//            if (j < mesh.nInternalFaces) {
//                int n = mesh.neighbour(0, j);
//                phiC(i, n) -= phi(i, j)*mesh.areas(0, j)/mesh.volumes(0, n);
//            }
//        }
//    }
//   return phiC;
//}

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

