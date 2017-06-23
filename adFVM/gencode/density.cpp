#define timeIntegrator euler

void equation(const vec& rho, const mat& rhoU, const vec& rhoE, vec& drho, mat& drhoU, vec& drhoE, scalar& objective, scalar& minDtc) {
    // make decision between 1 and 3 a template
    // optimal memory layout? combine everything?
    //cout << "c++: equation 1" << endl;
    const Mesh& mesh = *meshp;

    mat U(mesh.nCells);
    vec T(mesh.nCells);
    vec p(mesh.nCells);

    Function_primitive(mesh.nInternalCells, &rho(0), &rhoU(0), &rhoE(0), &U(0), &T(0), &p(0));

    //this->boundaryInit();    
    //this->boundary(this->boundaries[0], U);
    //this->boundary(this->boundaries[1], T);
    //this->boundary(this->boundaries[2], p);
    ////this->boundaryEnd();    
    ////U.info();
    ////T.info();
    ////p.info();

    arrType<scalar, 3, 3> gradU(mesh.nCells);
    arrType<scalar, 1, 3> gradT(mesh.nCells);
    arrType<scalar, 1, 3> gradp(mesh.nCells);
    gradU.zero();
    gradT.zero();
    gradp.zero();
    //
    #define gradUpdate(i, n, func) \
        func(n, \
                &U(0), &T(0), &p(0), \
                &mesh.areas(i), &mesh.volumesL(i), &mesh.volumesR(i), \
                &mesh.weights(i), &mesh.deltas(i), &mesh.normals(i), \
                &mesh.linearWeights(i), &mesh.quadraticWeights(i), \
                &mesh.owner(i), &mesh.neighbour(i), \
                &gradU(0), &gradT(0), &gradp(0));

    gradUpdate(0, mesh.nInternalFaces, Function_grad);
    //this->boundaryEnd();    
    for (auto& patch: mesh.boundary) {
        auto& patchInfo = patch.second;
        integer startFace, nFaces;
        tie(startFace, nFaces) = mesh.boundaryFaces.at(patch.first);
        //if (patchInfo.at("type") == "cyclic") {
        if ((patchInfo.at("type") == "cyclic") ||
            (patchInfo.at("type") == "processor") ||
            (patchInfo.at("type") == "processorCyclic")) {
            gradUpdate(startFace, nFaces, Function_coupledGrad);
        } else {
            gradUpdate(startFace, nFaces, Function_boundaryGrad);
        }
    }
    ////faceUpdate(mesh.nLocalFaces, mesh.nFaces, false);
    ////cout << "gradU " << gradU.checkNAN() << endl;
    //
    ////cout << "c++: equation 3" << endl;
    //this->boundaryInit(this->reqField);    
    //this->boundary(mesh.defaultBoundary, gradU);
    //this->boundary(mesh.defaultBoundary, gradT);
    //this->boundary(mesh.defaultBoundary, gradp);
    ////this->boundaryEnd();
    //
    vec dtc(mesh.nCells);
    drho.zero();
    drhoU.zero();
    drhoE.zero();
    dtc.zero();
    //objective = this->objective(this, U, T, p);

    #define fluxUpdate(i, n, func) \
        func(n, \
                &U(0), &T(0), &p(0), \
                &gradU(0), &gradT(0), &gradp(0), \
                &mesh.areas(i), &mesh.volumesL(i), &mesh.volumesR(i), \
                &mesh.weights(i), &mesh.deltas(i), &mesh.normals(i), \
                &mesh.linearWeights(i), &mesh.quadraticWeights(i), \
                &mesh.owner(i), &mesh.neighbour(i), \
                &drho(0), &drhoU(0), &drhoE(0));

    fluxUpdate(0, mesh.nInternalFaces, Function_flux);
    //this->boundaryEnd();    
    for (auto& patch: mesh.boundary) {
        auto& patchInfo = patch.second;
        integer startFace, nFaces;
        tie(startFace, nFaces) = mesh.boundaryFaces.at(patch.first);
        //if (patchInfo.at("type") == "cyclic") {
        if ((patchInfo.at("type") == "cyclic") ||
            (patchInfo.at("type") == "processor") ||
            (patchInfo.at("type") == "processorCyclic")) {
            fluxUpdate(startFace, nFaces, Function_coupledFlux);
        } else if (patchInfo.at("type") == "characteristic") {
            fluxUpdate(startFace, nFaces, Function_characteristicFlux);
        } else {
            fluxUpdate(startFace, nFaces, Function_boundaryFlux);
        }
    }

    //auto fluxUpdate(int startFace, int nFaces, auto func) {
    //   func(UL.data, )
    //}
    //auto [start]
   
    //Function_flux();
    //this->boundaryEnd();    
    ////cout << "c++: equation 4" << endl;
    //// characteristic boundary
    //for (auto& patch: mesh.boundary) {
    //    auto& patchInfo = patch.second;
    //    integer startFace, nFaces;
    //    tie(startFace, nFaces) = mesh.boundaryFaces.at(patch.first);
    //    integer cellStartFace = mesh.nInternalCells + startFace - mesh.nInternalFaces;
    //    string patchType = patchInfo.at("type");
    //    //if (patchInfo.at("type") == "cyclic") {
    //    if ((patchType == "cyclic") ||
    //        (patchType == "processor") ||
    //        (patchType == "processorCyclic")) {
    //        faceFluxUpdate(startFace, startFace + nFaces, false, false);
    //    } else if (patchType == "characteristic") {
    //        faceFluxUpdate(startFace, startFace + nFaces, false, true);
    //    } else {
    //        for (integer i = 0; i < nFaces; i++) {
    //            integer index = startFace + i;
    //            integer c = cellStartFace + i;
    //            scalar rhoFlux;
    //            scalar rhoUFlux[3];
    //            scalar rhoEFlux;

    //            this->getFlux(&U(c), T(c), p(c), &mesh.normals(index), rhoFlux, rhoUFlux, rhoEFlux);
    //            viscousFluxUpdate(&U(c), T(c), rhoUFlux, rhoEFlux, index);

    //            this->operate->div(&rhoFlux, drho, index, false);
    //            this->operate->div(rhoUFlux, drhoU, index, false);
    //            this->operate->div(&rhoEFlux, drhoE, index, false);

    //            scalar aF = sqrt((this->gamma-1)*this->Cp*T(c));
    //            scalar maxaF = 0;
    //            for (integer j = 0; j < 3; j++) {
    //                maxaF += U(c, j)*mesh.normals(i, j);
    //            }
    //            maxaF = fabs(maxaF) + aF;
    //            this->operate->absDiv(&maxaF, dtc, i, false);
    //        }
    //    }
    //}
    ////faceFluxUpdate(mesh.nLocalFaces, mesh.nFaces, false);
    ////cout << "c++: equation 5" << endl;
    ////
    //
    //// CFL computation
    //minDtc = 1e100;
    //for (integer i = 0; i < mesh.nInternalCells; i++) {
    //    minDtc = min(2*this->CFL/dtc(i), minDtc);
    //    drho(i) -= (*this->rhoS)(i);
    //    for (integer j = 0; j < 3; j++) {
    //        drhoU(i) -= (*this->rhoUS)(i, j);
    //    }
    //    drhoE(i) -= (*this->rhoES)(i);
    //}
    //cout << minDtc << endl;
}

tuple<scalar, scalar> euler(const vec& rho, const mat& rhoU, const vec& rhoE, vec& rhoN, mat& rhoUN, vec& rhoEN, scalar t, scalar dt) {
    const Mesh& mesh = *meshp;
    
    vec drho(rho.shape);
    mat drhoU(rhoU.shape);
    vec drhoE(rhoE.shape);
    scalar objective, dtc;
    equation(rho, rhoU, rhoE, drho, drhoU, drhoE, objective, dtc);

    for (integer i = 0; i < mesh.nInternalCells; i++) {
        rhoN(i) = rho(i) - dt*drho(i);
        for (integer j = 0; j < 3; j++) {
            rhoUN(i, j) = rhoU(i, j) - dt*drhoU(i, j);
        }
        rhoEN(i) = rhoE(i) - dt*drhoE(i);
    }
    return make_tuple(objective, dtc);
}

tuple<scalar, scalar> SSPRK(const vec& rho, const mat& rhoU, const vec& rhoE, vec& rhoN, mat& rhoUN, vec& rhoEN, scalar t, scalar dt) {
    const Mesh& mesh = *meshp;

    const integer n = 3;
    scalar alpha[n][n] = {{1,0,0},{3./4, 1./4, 0}, {1./3, 0, 2./3}};
    scalar beta[n][n] = {{1,0,0}, {0,1./4,0},{0,0,2./3}};
    scalar gamma[n] = {0, 1, 0.5};
    scalar objective[n], dtc[n];

    vec rhos[n+1] = {{rho.shape, rho.data}, {rho.shape}, {rho.shape}, {rho.shape, rhoN.data}};
    mat rhoUs[n+1] = {{rhoU.shape, rhoU.data}, {rhoU.shape}, {rhoU.shape}, {rhoU.shape, rhoUN.data}};
    vec rhoEs[n+1] = {{rhoE.shape, rhoE.data}, {rhoE.shape}, {rhoE.shape}, {rhoE.shape, rhoEN.data}};
    vec drho(rho.shape);
    mat drhoU(rhoU.shape);
    vec drhoE(rhoE.shape);

    for (integer stage = 0; stage < n; stage++) {
        //solver.t = solver.t0 + gamma[i]*solver.dt
        //rcf->stage = stage;
        equation(rhos[stage], rhoUs[stage], rhoEs[stage], drho, drhoU, drhoE, objective[stage], dtc[stage]);
        integer curr = stage + 1;
        scalar b = beta[stage][stage];
        for (integer i = 0; i < mesh.nInternalCells; i++) {
            rhos[curr](i) = -b*drho(i)*dt;
            for (integer j = 0; j < 3; j++) {
                rhoUs[curr](i, j) = -b*drhoU(i, j)*dt;
            }
            rhoEs[curr](i) = -b*drhoE(i)*dt;
        }
        for (integer prev = 0; prev < curr; prev++) {
            scalar a = alpha[stage][prev];
            for (integer i = 0; i < mesh.nInternalCells; i++) {
                rhos[curr](i) += a*rhos[prev](i);
                for (integer j = 0; j < 3; j++) {
                    rhoUs[curr](i, j) += a*rhoUs[prev](i, j);
                }
                rhoEs[curr](i) += a*rhoEs[prev](i);
            }
        }
    }
    return make_tuple(objective[0], dtc[0]);
}



