/*void RCF::boundaryUPT_grad(mat& U, vec& T, vec& p) {*/
    //const Mesh& mesh = *meshp;
    //for (auto& patch: this->boundaries[2]) {
        //string patchType = patch.second.at("type");
        //string patchID = patch.first;
        //integer startFace, nFaces;
        //tie(startFace, nFaces) = mesh.boundaryFaces.at(patchID);
        //integer cellStartFace = mesh.nInternalCells + startFace - mesh.nInternalFaces;

        //if (patchType == "CBC_UPT") {
            //mat Uval(nFaces, patch.second.at("_U0"));
            //vec Tval(nFaces, patch.second.at("_T0"));
            //vec pval(nFaces, patch.second.at("_p0"));
            //for (integer i = 0; i < nFaces; i++) {
                //integer c = cellStartFace + i;
                //for (integer j = 0; j < 3; j++) {
                    //U(c, j) = Uval(i, j);
                //}
                //T(c) = Tval(i);
                //p(c) = pval(i);
            //}
        //} else if (patchType == "CBC_TOTAL_PT") {
            //cout << "implement this" << endl;
        //}
    //}
    //this->boundaryInit(0);    
    //boundary(this->boundaries[0], U);
    //boundary(this->boundaries[1], T);
    //boundary(this->boundaries[2], p);
    //this->boundaryEnd();    
/*}*/

void RCF::equation_grad(const vec& rho, const mat& rhoU, const vec& rhoE, const vec& drhoa, const mat& drhoUa, const vec& drhoEa, vec& rhoa, mat& rhoUa, vec& rhoEa, scalar& objective, scalar& minDtc) {
    // make decision between 1 and 3 a template
    // optimal memory layout? combine everything?
    //cout << "c++: equation 1" << endl;
    const Mesh& mesh = *meshp;
    Mesh& meshAdj = *meshap;

    integer index = this->stage;

    const mat& U = *Us[index];
    const vec& T = *Ts[index];
    const vec& p = *ps[index];
    const arrType<scalar, 3, 3>& gradU = *gradUs[index];
    const arrType<scalar, 1, 3>& gradT = *gradTs[index];
    const arrType<scalar, 1, 3>& gradp = *gradps[index];

    mat Ua(mesh.nCells, true);
    vec Ta(mesh.nCells, true);
    vec pa(mesh.nCells, true);
    
    arrType<scalar, 3, 3> gradUa(mesh.nCells, true);
    arrType<scalar, 1, 3> gradTa(mesh.nCells, true);
    arrType<scalar, 1, 3> gradpa(mesh.nCells, true);

    // extra burden
    vec dtca(mesh.nInternalCells, true);

    #define meshInputs(meshType, i) \
                &meshType.areas(i), &meshType.volumesL(i), &meshType.volumesR(i), \
                &meshType.weights(i), &meshType.deltas(i), &meshType.normals(i), \
                &meshType.linearWeights(i), &meshType.quadraticWeights(i), \
                &meshType.owner(i), &meshType.neighbour(i)

    #define gradFluxUpdate(i, n, func) \
                func(n, \
                &U(0), &T(0), &p(0), \
                &gradU(0), &gradT(0), &gradp(0), \
                meshInputs(mesh, i), \
                &drhoa(0), &drhoUa(0), &drhoEa(0), &dtca(0), \
                &Ua(0), &Ta(0), &pa(0), \
                &gradUa(0), &gradTa(0), &gradpa(0), \
                meshInputs(meshAdj, i) \
                );

    gradFluxUpdate(0, mesh.nInternalFaces, Function_flux_grad);
    //this->boundaryEnd();    
    for (auto& patch: mesh.boundary) {
        auto& patchInfo = patch.second;
        integer startFace, nFaces;
        tie(startFace, nFaces) = mesh.boundaryFaces.at(patch.first);
        //if (patchInfo.at("type") == "cyclic") {
        if ((patchInfo.at("type") == "cyclic") ||
            (patchInfo.at("type") == "processor") ||
            (patchInfo.at("type") == "processorCyclic")) {
            gradFluxUpdate(startFace, nFaces, Function_coupledFlux_grad);
        } else if (patchInfo.at("type") == "characteristic") {
            gradFluxUpdate(startFace, nFaces, Function_characteristicFlux_grad);
        } else {
            gradFluxUpdate(startFace, nFaces, Function_boundaryFlux_grad);
        }
    }
    
    // grad BC
    this->boundaryInit(0);
    this->boundary_grad(mesh.defaultBoundary, gradUa);
    this->boundary_grad(mesh.defaultBoundary, gradTa);
    this->boundary_grad(mesh.defaultBoundary, gradpa);
    this->boundaryEnd();
    this->boundaryEnd_grad(gradUa, this->reqBuf[0]);
    this->boundaryEnd_grad(gradTa, this->reqBuf[1]);
    this->boundaryEnd_grad(gradpa, this->reqBuf[2]);

    #define gradGradUpdate(i, n, func) \
                func(n, \
                &U(0), &T(0), &p(0), \
                meshInputs(mesh, i), \
                &gradUa(0), &gradTa(0), &gradpa(0), \
                &Ua(0), &Ta(0), &pa(0), \
                meshInputs(meshAdj, i) \
                );

    gradGradUpdate(0, mesh.nInternalFaces, Function_grad_grad);
    //this->boundaryEnd();    
    for (auto& patch: mesh.boundary) {
        auto& patchInfo = patch.second;
        integer startFace, nFaces;
        tie(startFace, nFaces) = mesh.boundaryFaces.at(patch.first);
        //if (patchInfo.at("type") == "cyclic") {
        if ((patchInfo.at("type") == "cyclic") ||
            (patchInfo.at("type") == "processor") ||
            (patchInfo.at("type") == "processorCyclic")) {
            gradGradUpdate(startFace, nFaces, Function_coupledGrad_grad);
        } else {
            gradGradUpdate(startFace, nFaces, Function_boundaryGrad_grad);
        }
    }

    // UPT BC
    this->boundaryInit(this->reqField);
    this->boundary_grad(this->boundaries[0], Ua);
    this->boundary_grad(this->boundaries[1], Ta);
    this->boundary_grad(this->boundaries[2], pa);
    this->boundaryEnd();
    this->boundaryEnd_grad(Ua, this->reqBuf[3]);
    this->boundaryEnd_grad(Ta, this->reqBuf[4]);
    this->boundaryEnd_grad(pa, this->reqBuf[5]);
    // CBC UPT
    
    // Primitive
    Function_primitive_grad(mesh.nInternalCells, &rho(0), &rhoU(0), &rhoE(0), &Ua(0), &Ta(0), &pa(0), \
                                                 &rhoa(0), &rhoUa(0), &rhoEa(0));
    
}

tuple<scalar, scalar> SSPRK_grad(const vec& rho, const mat& rhoU, const vec& rhoE, const vec& rhoa, const mat& rhoUa, const vec& rhoEa, vec& rhoaN, mat& rhoUaN, vec& rhoEaN, scalar t, scalar dt) {
    const Mesh& mesh = *meshp;

    integer n = nStages;
    scalar alpha[n][n] = {{1,0,0},{3./4, 1./4, 0}, {1./3, 0, 2./3}};
    scalar beta[n][n] = {{1,0,0}, {0,1./4,0},{0,0,2./3}};
    //scalar gamma[n] = {0, 1, 0.5};
    scalar objective[n], dtc[n];

    vec rhoas[n+1] = {{rho.shape, rhoaN.data}, {rho.shape}, {rho.shape}, {rho.shape, rhoa.data}};
    mat rhoUas[n+1] = {{rhoU.shape, rhoUaN.data}, {rhoU.shape}, {rhoU.shape}, {rhoU.shape, rhoUa.data}};
    vec rhoEas[n+1] = {{rhoE.shape, rhoEaN.data}, {rhoE.shape}, {rhoE.shape}, {rhoE.shape, rhoEa.data}};

    vec drhoa(rho.shape);
    mat drhoUa(rhoU.shape);
    vec drhoEa(rhoE.shape);

    for (integer stage = n; stage > 0; stage--) {
        //solver.t = solver.t0 + gamma[i]*solver.dt
        integer curr = stage - 1;
        rcf->stage = curr;
        scalar b = beta[curr][curr];
        for (integer i = 0; i < mesh.nInternalCells; i++) {
            drhoa(i) = -b*rhoas[stage](i)*dt;
            for (integer j = 0; j < 3; j++) {
                drhoUa(i) = -b*rhoUas[stage](i, j)*dt;
            }
            drhoEa(i) = -b*rhoEas[stage](i)*dt;
        }
        rcf->equation_grad(*rhos[curr], *rhoUs[curr], *rhoEs[curr], drhoa, drhoUa, drhoEa, rhoas[curr], rhoUas[curr], rhoEas[curr], objective[stage], dtc[stage]);
        for (integer prev = stage; prev <= n; prev++) {
            scalar a = alpha[prev-1][curr];
            for (integer i = 0; i < mesh.nInternalCells; i++) {
                rhoas[curr](i) += a*rhoas[prev](i);
                for (integer j = 0; j < 3; j++) {
                    rhoUas[curr](i, j) += a*rhoUas[prev](i, j);
                }
                rhoEas[curr](i) += a*rhoEas[prev](i);
            }
        }
    }
    return make_tuple(objective[0], dtc[0]);
}

template <typename dtype, integer shape1, integer shape2>
void RCF::boundary_grad(const Boundary& boundary, arrType<dtype, shape1, shape2>& phi) {
    const Mesh& mesh = *meshp;
    //MPI_Barrier(MPI_COMM_WORLD);

    dtype *phiBuf = NULL; 
    integer reqPos = 0;
    if (mesh.nRemotePatches > 0) {
        reqPos = this->reqIndex/(2*mesh.nRemotePatches);
        phiBuf = new dtype[(mesh.nCells-mesh.nLocalCells)*shape1*shape2];
        this->reqBuf[reqPos] = phiBuf;
    }

    for (auto& patch: boundary) {
        string patchType = patch.second.at("type");
        string patchID = patch.first;
        const map<string, string>& patchInfo = mesh.boundary.at(patchID);

        integer startFace, nFaces;
        tie(startFace, nFaces) = mesh.boundaryFaces.at(patch.first);
        integer cellStartFace = mesh.nInternalCells + startFace - mesh.nInternalFaces;

        if (patchType == "cyclic") {
            string neighbourPatchID = patchInfo.at("neighbourPatch");
            integer neighbourStartFace = std::get<0>(mesh.boundaryFaces.at(neighbourPatchID));
            for (integer i = 0; i < nFaces; i++) {
                integer p = mesh.owner(neighbourStartFace + i);
                integer c = cellStartFace + i;
                for (integer j = 0; j < shape1; j++) {
                    for (integer k = 0; k < shape2; k++) {
                        phi(p, j, k) += phi(c, j, k);
                    }
                }
            }
        } else if (patchType == "zeroGradient" || patchType == "empty" || patchType == "inletOutlet") {
            for (integer i = 0; i < nFaces; i++) {
                integer p = mesh.owner(startFace + i);
                integer c = cellStartFace + i;
                for (integer j = 0; j < shape1; j++) {
                    for (integer k = 0; k < shape2; k++) {
                        phi(p, j, k) += phi(c, j, k);
                    }
                }
            }
        } else if (patchType == "symmetryPlane" || patchType == "slip") {
            //cout << "implement this elsewhere" << endl;
            //if ((shape1 == 3) && (shape2 == 1)) {
            //    for (integer i = 0; i < nFaces; i++) {
            //        integer f = startFace + i;
            //        integer c = cellStartFace + i;
            //        integer p = mesh.owner(f);
            //        dtype phin = 0.;
            //        for (integer j = 0; j < 3; j++) {
            //            phin += mesh.normals(f, j)*phi(p, j);
            //        }
            //        for (integer j = 0; j < 3; j++) {
            //            phi(c, j) = phi(p, j) - mesh.normals(f, j)*phin;
            //        }
            //    }
            //} else {
            //    for (integer i = 0; i < nFaces; i++) {
            //        integer p = mesh.owner(startFace + i);
            //        integer c = cellStartFace + i;
            //        for (integer j = 0; j < shape1; j++) {
            //            for (integer k = 0; k < shape2; k++) {
            //                phi(c, j, k) = phi(p, j, k);
            //            }
            //        }
            //    }
            //}
        } else if (patchType == "fixedValue") {
            //arrType<scalar, shape1, shape2> phiVal(nFaces, patch.second.at("_value"));

            //for (integer i = 0; i < nFaces; i++) {
            //    integer c = cellStartFace + i;
            //    for (integer j = 0; j < shape1; j++) {
            //        for (integer k = 0; k < shape2; k++) {
            //            phi(c, j, k) = phiVal(i, j, k);
            //        }
            //    }
            //}
        } else if (patchType == "processor" || patchType == "processorCyclic") {
            //cout << "hello " << patchID << endl;
            integer bufStartFace = cellStartFace - mesh.nLocalCells;
            integer size = nFaces*shape1*shape2;
            integer dest = stoi(patchInfo.at("neighbProcNo"));
            
            MPI_Request *req = (MPI_Request*) this->req;
            integer tag = (this->stage*1000+1) + this->reqField*100 + mesh.tags.at(patchID);
            //cout << patchID << " " << tag << endl;
            MPI_Isend(&phi(cellStartFace), size, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD, &req[this->reqIndex]);
            MPI_Irecv(&phiBuf[bufStartFace*shape1*shape2], size, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD, &req[this->reqIndex+1]);
            this->reqIndex += 2;
        }
        else if (patchType == "calculated") {
        } 
        //else {
        //    cout << "patch not found " << patchType << " for " << patchID << endl;
        //}
    }
    this->reqField++;
}

template <typename dtype, integer shape1, integer shape2>
void RCF::boundaryEnd_grad(arrType<dtype, shape1, shape2>& phi, dtype* phiBuf) {
    const Mesh& mesh = *meshp;
    //MPI_Barrier(MPI_COMM_WORLD);

    for (auto& patch: mesh.boundary) {
        string patchType = patch.second.at("type");
        string patchID = patch.first;
        integer startFace, nFaces;
        tie(startFace, nFaces) = mesh.boundaryFaces.at(patch.first);
        integer cellStartFace = mesh.nInternalCells + startFace - mesh.nInternalFaces;
        if (patchType == "processor" || patchType == "processorCyclic") {
            //cout << "hello " << patchID << endl;
            integer bufStartFace = cellStartFace - mesh.nLocalCells;
            for (integer i = 0; i < nFaces; i++) {
                integer p = mesh.owner(startFace + i);
                integer b = bufStartFace + i;
                for (integer j = 0; j < shape1; j++) {
                    for (integer k = 0; k < shape2; k++) {
                        phi(p, j, k) += phiBuf[b*shape1*shape2 + j*shape2 + k];
                    }
                }
            }
        }
    }
}

//void RCF::boundaryInit_grad(integer startField) {
    //const Mesh& mesh = *meshp;
    //this->reqIndex = 0;
    //this->reqField = startField;
    //if (mesh.nRemotePatches > 0) {
        ////MPI_Barrier(MPI_COMM_WORLD);
        //this->req = (void *)new MPI_Request[2*3*mesh.nRemotePatches];
    //}
//}
//
//

//void RCF::boundaryEnd_grad() {
    //const Mesh& mesh = *meshp;
    //if (mesh.nRemotePatches > 0) {
        //MPI_Waitall(2*3*mesh.nRemotePatches, ((MPI_Request*)this->req), MPI_STATUSES_IGNORE);
        //delete[] ((MPI_Request*)this->req);
        ////MPI_Barrier(MPI_COMM_WORLD);
        //for (integer i = 0; i < 3; i++) {
            //delete[] this->reqBuf[i];
        //}
    //}
//}

