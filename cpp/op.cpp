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
    arr gradF(phi.rows()*mesh.normals.rows(), phi.cols());
    for (int i = 0; i < phi.cols(); i++) {
        MatrixXd A = mesh.normals.col(i).matrix() * phi.col(i).matrix().transpose();
        VectorXd B(Map<VectorXd>(A.data(), A.cols()*A.rows()));
        gradF.col(i) = B;
    }
    return internal_sum(gradF);
}

arr Operator::div(const arr& phi) {
    return internal_sum(phi);
}

arr Operator::snGrad(const arr& phi) {
    arr phiN = slice(phi, mesh.neighbour);
    arr phiP = slice(phi, mesh.owner);
    arr gradFdotN = ROWMUL(phiN-phiP, mesh.deltas);
    return gradFdotN;
}

arr Operator::laplacian(const arr& phi) {
    return internal_sum(snGrad(phi));
}

