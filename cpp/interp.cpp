#include "interp.hpp"

arr Interpolator::central(const arr& phi) {
    arr phiN = slice(phi, mesh.neighbour);
    arr phiP = slice(phi, mesh.owner);
    return ROWMUL(phiN, 1-mesh.weights) + ROWMUL(phiP, mesh.weights);
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
        arr r;
        // compile time switch? not possible in current code structure
        if (phi.rows() == 1) {
            r = 2*DOT(gradC, R)/(gradF + SMALL) - 1;
        }
        else {
            //TODO
            r = 2*DOT(gradC, R)/(gradF + SMALL) - 1;
        }
        SELECT(phiF, startFace, nFaces) = phiC + 0.5*psi(r, r.abs())*gradF;
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
