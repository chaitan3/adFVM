#include "interp.hpp"

//arr Interpolator::central(const arr& phi) {
//    arr phiN = slice(phi, mesh.neighbour);
//    arr phiP = slice(phi, mesh.owner);
//    return ROWMUL(phiN, 1-mesh.weights) + ROWMUL(phiP, mesh.weights);
//}

arr Interpolator::central(const arr& phi) {
    arr phiF = arr::Zero(phi.rows(), mesh.nFaces);
    for (int j = 0; j < mesh.nFaces; j++) {
        for (int i = 0; i < phi.rows(); i++) {
            int p = mesh.owner(0, j);
            int n = mesh.neighbour(0, j);
            int w = mesh.weights(0, j);
            phiF(i, j) = phi(i, n)*(1-w) + phi(i, p)*w;
        }
    }
    return phiF;
}


// maybe no lambda funcs?
arr Interpolator::TVD(const arr& phi, const arr& gradPhi, const arr& U) {
    arr phiF(phi.rows(), mesh.nFaces);

    auto psi = [](const arr& r, const arr& rabs) {
        return (r + rabs)/(1 + rabs);
    };

    auto update = [&](int startFace, int nFaces) {
        iarr C = (SELECT(U, startFace, nFaces) > 0).select(SELECT(mesh.owner, startFace, nFaces), SELECT(mesh.neighbour, startFace, nFaces));
        iarr D = (SELECT(U, startFace, nFaces) > 0).select(SELECT(mesh.neighbour, startFace, nFaces), SELECT(mesh.owner, startFace, nFaces));
        arr phiC = slice(phi, C);
        arr gradF = slice(phi, D) - phiC;
        arr R = slice(mesh.cellCentres, D) - slice(mesh.cellCentres, C);
        arr gradC = slice(gradPhi, C);
        
        //arr phiC(phi.rows(), nFaces);
        //arr gradF(phi.rows(), nFaces);
        //arr R(mesh.cellCentres.rows(), nFaces);
        //arr gradC(gradPhi.rows(), nFaces);

        //for (int j = startFace; j < nFaces; j++) {
        //    int C, D;
        //    if (U(0, j) > 0) {
        //        C = mesh.owner(0, j);
        //        D = mesh.neighbour(0, j);
        //    } else {
        //        C = mesh.owner(0, j);
        //        D = mesh.neighbour(0, j);
        //    }
        //    for (int i = 0; i < phi.rows(); i++) {
        //       phiC(i, j) = phi(i, C);
        //       gradF(i, j) = phi(i, D) - phi(i, C);
        //    }
        //    for (int i = 0; i < mesh.cellCentres.rows(); i++) {
        //       R(i, j) = mesh.cellCentres(i, D) - mesh.cellCentres(i, C);
        //    }
        //    for (int i = 0; i < gradPhi.rows(); i++) {
        //       gradC(i, j) = gradPhi(i, C);
        //    }
        //}

        arr r;
        //// compile time switch? not possible in current code structure
        if (phi.rows() == 1) {
            r = 2*DOT(gradC, R)/stabilise(gradF, SMALL) - 1;
        }
        else {
            r = 2*DOT(tdot(transpose(gradC), R), gradF)/stabilise(DOT(gradF, gradF),  SMALL) - 1;
        }
        SELECT(phiF, startFace, nFaces) = phiC + 0.5*ROWMUL(gradF, psi(r, r.abs()));

    };

    update(0, mesh.nInternalFaces);
    for (auto &patch: mesh.boundary) {
        int startFace = stoi(patch.second.at("startFace"));
        int nFaces = stoi(patch.second.at("nFaces"));
        string patchType = patch.second.at("type");
        if (patchType == "cyclic") {
            update(startFace, nFaces);
        }
        else {
            SELECT(phiF, startFace, nFaces) = slice(phi, SELECT(mesh.neighbour, startFace, nFaces));
        }
    }

    return phiF;
}
