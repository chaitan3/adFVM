#include "density.hpp"
#include "riemann.hpp"
#ifdef ADIFF
    #include "externals/ampi_interface_realreverse.cpp"
#endif

//// confirm that make_tuple doesn't create copies
void RCF::primitive(const scalar rho, const scalar rhoU[3], const scalar rhoE, scalar U[3], scalar& T, scalar& p) {
    scalar U2 = 0.;
    for (integer i = 0; i < 3; i++) {
        U[i] = rhoU[i]/rho;
        U2 += U[i]*U[i];
    }
    scalar e = rhoE/rho - 0.5*U2;
    T = e/this->Cv, 
    p = (this->gamma-1)*rho*e;
}

void RCF::conservative(const scalar U[3], const scalar T, const scalar p, scalar& rho, scalar rhoU[3], scalar& rhoE) {
    scalar e = this->Cv * T;
    rho = p/(e*(this->gamma - 1));
    scalar U2 = 0.;
    for (integer i = 0; i < 3; i++) {
        U2 += U[i]*U[i];
        rhoU[i] = rho*U[i];
    }
    rhoE = rho*(e + 0.5*U2);
}


void RCF::getFlux(const scalar U[3], const scalar T, const scalar p, const uscalar N[3], scalar& rhoFlux, scalar rhoUFlux[3], scalar& rhoEFlux) {
    scalar rho, rhoU[3], rhoE;
    this->conservative(U, T, p, rho, rhoU, rhoE);
    scalar Un = 0.;
    for (integer i = 0; i < 3; i++) {
        Un += U[i]*N[i];
    }
    rhoFlux = rho*Un;
    for (integer i = 0; i < 3; i++) {
        rhoUFlux[i] = rhoU[i]*Un + p*N[i];
    }
    rhoEFlux = (rhoE + p)*Un;
}

void RCF::equation(const vec& rho, const mat& rhoU, const vec& rhoE, vec& drho, mat& drhoU, vec& drhoE, scalar& objective, scalar& minDtc) {
    // make decision between 1 and 3 a template
    // optimal memory layout? combine everything?
    //cout << "c++: equation 1" << endl;
    const Mesh& mesh = *this->mesh;

    mat U(mesh.nCells);
    vec T(mesh.nCells);
    vec p(mesh.nCells);
    for (integer i = 0; i < mesh.nInternalCells; i++) {
        this->primitive(rho(i), &rhoU(i), rhoE(i), &U(i), T(i), p(i));
    }
    //cout << "c++: equation 2" << endl;

    this->U = &U;
    this->T = &T;
    this->p = &p;
    this->boundaryInit();    
    this->boundary(this->boundaries[0], U);
    this->boundary(this->boundaries[1], T);
    this->boundary(this->boundaries[2], p);
    //this->boundaryEnd();    
    //U.info();
    //T.info();
    //p.info();

    arrType<scalar, 3, 3> gradU(mesh.nCells);
    arrType<scalar, 1, 3> gradT(mesh.nCells);
    arrType<scalar, 1, 3> gradp(mesh.nCells);
    gradU.zero();
    gradT.zero();
    gradp.zero();


    
    auto faceUpdate = [&](const integer start, const integer end, const bool neighbour) {
        for (integer i = start; i < end; i++) {
            scalar UF[3], TF, pF;
            this->interpolate->central(U, UF, i);
            this->interpolate->central(T, &TF, i);
            this->interpolate->central(p, &pF, i);
            this->operate->grad(UF, gradU, i, neighbour);
            this->operate->grad(&TF, gradT, i, neighbour);
            this->operate->grad(&pF, gradp, i, neighbour);
        }
        //cout << start << " " << U.checkNAN() << endl;
        //cout << end << " " << gradT.checkNAN() << endl;
    };
    faceUpdate(0, mesh.nInternalFaces, true);
    this->boundaryEnd();    
    for (auto& patch: mesh.boundary) {
        auto& patchInfo = patch.second;
        integer startFace, nFaces;
        tie(startFace, nFaces) = mesh.boundaryFaces.at(patch.first);
        integer cellStartFace = mesh.nInternalCells + startFace - mesh.nInternalFaces;
        //if (patchInfo.at("type") == "cyclic") {
        if ((patchInfo.at("type") == "cyclic") ||
            (patchInfo.at("type") == "processor") ||
            (patchInfo.at("type") == "processorCyclic")) {
            faceUpdate(startFace, startFace + nFaces, false);
        } else {
            for (integer i = 0; i < nFaces; i++) {
                integer index = startFace + i;
                integer c = cellStartFace + i;
                this->operate->grad(&U(c), gradU, index, false);
                this->operate->grad(&T(c), gradT, index, false);
                this->operate->grad(&p(c), gradp, index, false);
            }
        }
    }
    //faceUpdate(mesh.nLocalFaces, mesh.nFaces, false);
    //cout << "gradU " << gradU.checkNAN() << endl;
    
    //cout << "c++: equation 3" << endl;
    this->boundaryInit(this->reqField);    
    this->boundary(mesh.defaultBoundary, gradU);
    this->boundary(mesh.defaultBoundary, gradT);
    this->boundary(mesh.defaultBoundary, gradp);
    //this->boundaryEnd();    
    
    vec dtc(mesh.nCells);
    drho.zero();
    drhoU.zero();
    drhoE.zero();
    dtc.zero();
    objective = this->objective(this, U, T, p);
    ///cout << std::setprecision (std::numeric_limits<double>::digits10 + 1) << objective << endl;

    /*auto viscousFluxUpdate = [&](const scalar UF[3], const scalar TF, scalar rhoUFlux[3], scalar& rhoEFlux, integer ind) {*/
        //scalar qF = 0, sigmadotUF = 0., sigmaF[3];
        //scalar mu = (this->*(this->mu))(TF);
        ////cout << mu << endl;
        //scalar kappa = this->kappa(mu, TF);

        //scalar gradTF[3], gradTCF[3];
        //scalar snGradT, gradTFs = 0.;
        //const uscalar* S = &mesh.deltasUnit(ind);
        //const uscalar* N = &mesh.normals(ind);
        //this->operate->snGrad(T, &snGradT, ind);
        ////this->interpolate->central(gradT, gradTF, ind);
        //this->interpolate->average(gradT, gradTF, ind);
        //for (integer i = 0; i < 3; i++) {
            //gradTFs += gradTF[i]*S[i];
        //}
        //for (integer i = 0; i < 3; i++) {
            //gradTCF[i] = gradTF[i] + snGradT*S[i] - gradTFs*S[i];
            //qF += kappa*(gradTCF[i]*N[i]);
        //}

        //scalar gradUF[3][3], gradUCF[3][3];
        //scalar snGradU[3];
        //this->interpolate->average(gradU, (scalar*)gradUF, ind);
        ////this->interpolate->central(gradU, (scalar*)gradUF, ind);
        //this->operate->snGrad(U, snGradU, ind);

        //scalar tmp[3], tmp2[3], tmp3;
        //for (integer i = 0; i < 3; i++) {
            //tmp[i] = 0;
            //for (integer j = 0; j < 3; j++) {
                //tmp[i] += gradUF[i][j]*S[j];
            //}
        //}
        //for (integer i = 0; i < 3; i++) {
            //for (integer j = 0; j < 3; j++) {
                //gradUCF[i][j] = gradUF[i][j] + snGradU[i]*S[j] - tmp[i]*S[j];
            //}
        //}
        //tmp3 = 0;
        //for (integer i = 0; i < 3; i++) {
            //tmp2[i] = 0;
            //for (integer j = 0; j < 3; j++) {
                //tmp2[i] += (gradUCF[i][j] + gradUCF[j][i])*N[j];
            //}
            //tmp3 += gradUCF[i][i];
        //}
        //for (integer i = 0; i < 3; i++) {
            //sigmaF[i] = mu*(tmp2[i] - 2./3*tmp3*N[i]);
            //rhoUFlux[i] -= sigmaF[i];
            //sigmadotUF += sigmaF[i]*UF[i];
        //}
        //rhoEFlux -= qF + sigmadotUF;
    /*};*/
    auto viscousFluxUpdate = [&](const scalar UF[3], const scalar TF, scalar rhoUFlux[3], scalar& rhoEFlux, integer ind) {
        scalar qF = 0, sigmadotUF = 0., sigmaF[3];
        scalar mu = (this->*(this->mu))(TF);
        //cout << mu << endl;
        scalar kappa = this->kappa(mu, TF);

        scalar gradTF[3];
        scalar snGradT;
        //const uscalar* S = &mesh.deltasUnit(ind);
        const uscalar* N = &mesh.normals(ind);
        this->operate->snGrad(T, &snGradT, ind);
        //this->interpolate->central(gradT, gradTF, ind);
        this->interpolate->average(gradT, gradTF, ind);
        qF += kappa*snGradT;

        scalar gradUF[3][3];
        //scalar snGradU[3];
        this->interpolate->average(gradU, (scalar*)gradUF, ind);
        //this->interpolate->central(gradU, (scalar*)gradUF, ind);
        //this->operate->snGrad(U, snGradU, ind);
        scalar tmp2[3], tmp3 = 0.;
        for (integer i = 0; i < 3; i++) {
            tmp2[i] = 0;
            for (integer j = 0; j < 3; j++) {
                tmp2[i] += (gradUF[i][j] + gradUF[j][i])*N[j];
            }
            tmp3 += gradUF[i][i];
        }
        for (integer i = 0; i < 3; i++) {
            sigmaF[i] = mu*(tmp2[i] - 2./3*tmp3*N[i]);
            rhoUFlux[i] -= sigmaF[i];
            sigmadotUF += sigmaF[i]*UF[i];
        }
        rhoEFlux -= qF + sigmadotUF;
    };
    auto faceFluxUpdate = [&](const integer start, const integer end, const bool neighbour, const bool characteristic) {
        for (integer i = start; i < end; i++) {
            scalar ULF[3], URF[3];
            scalar TLF, TRF;
            scalar pLF, pRF;
            
            this->interpolate->faceReconstructor(U, gradU, ULF, i, 0);
            this->interpolate->faceReconstructor(T, gradT, &TLF, i, 0);
            this->interpolate->faceReconstructor(p, gradp, &pLF, i, 0);
            if (characteristic) {
                integer c = mesh.nInternalCells + i - mesh.nInternalFaces;
                for (integer j = 0; j < 3; j++) {
                    URF[j] = U(c, j);
                }
                TRF = T(c);
                pRF = p(c);
            } else {
                this->interpolate->faceReconstructor(U, gradU, URF, i, 1);
                this->interpolate->faceReconstructor(T, gradT, &TRF, i, 1);
                this->interpolate->faceReconstructor(p, gradp, &pRF, i, 1);
            }

            scalar rhoLF, rhoRF;
            scalar rhoULF[3], rhoURF[3];
            scalar rhoELF, rhoERF;
            this->conservative(ULF, TLF, pLF, rhoLF, rhoULF, rhoELF);
            this->conservative(URF, TRF, pRF, rhoRF, rhoURF, rhoERF);
            scalar rhoFlux;
            scalar rhoUFlux[3];
            scalar rhoEFlux;
            if (characteristic) {
                boundaryRiemannSolver(this->gamma, \
                pLF, pRF, TLF, TRF, ULF, URF, \
                rhoLF, rhoRF, rhoULF, rhoURF, rhoELF, rhoERF, &mesh.normals(i), 
                rhoFlux, rhoUFlux, rhoEFlux);
            } else {
                riemannSolver(this->gamma, \
                pLF, pRF, TLF, TRF, ULF, URF, \
                rhoLF, rhoRF, rhoULF, rhoURF, rhoELF, rhoERF, &mesh.normals(i), 
                rhoFlux, rhoUFlux, rhoEFlux);
            }
            scalar UF[3], TF;
            for (integer j = 0; j < 3; j++) {
                UF[j] = 0.5*(ULF[j] + URF[j]);
            }
            TF = 0.5*(TLF + TRF);
            viscousFluxUpdate(UF, TF, rhoUFlux, rhoEFlux, i);

            this->operate->div(&rhoFlux, drho, i, neighbour);
            this->operate->div(rhoUFlux, drhoU, i, neighbour);
            this->operate->div(&rhoEFlux, drhoE, i, neighbour);

            scalar aF = sqrt((this->gamma-1)*this->Cp*TF);
            scalar maxaF = 0;
            for (integer j = 0; j < 3; j++) {
                maxaF += UF[j]*mesh.normals(i, j);
            }
            maxaF = fabs(maxaF) + aF;
            this->operate->absDiv(&maxaF, dtc, i, neighbour);
        }
        //cout << start << " " << drho.checkNAN() << endl;
        //cout << end << " " << drhoU.checkNAN() << endl;
    };
    faceFluxUpdate(0, mesh.nInternalFaces, true, false);
    this->boundaryEnd();    
    //cout << "c++: equation 4" << endl;
    // characteristic boundary
    for (auto& patch: mesh.boundary) {
        auto& patchInfo = patch.second;
        integer startFace, nFaces;
        tie(startFace, nFaces) = mesh.boundaryFaces.at(patch.first);
        integer cellStartFace = mesh.nInternalCells + startFace - mesh.nInternalFaces;
        string patchType = patchInfo.at("type");
        //if (patchInfo.at("type") == "cyclic") {
        if ((patchType == "cyclic") ||
            (patchType == "processor") ||
            (patchType == "processorCyclic")) {
            faceFluxUpdate(startFace, startFace + nFaces, false, false);
        } else if (patchType == "characteristic") {
            faceFluxUpdate(startFace, startFace + nFaces, false, true);
        } else {
            for (integer i = 0; i < nFaces; i++) {
                integer index = startFace + i;
                integer c = cellStartFace + i;
                scalar rhoFlux;
                scalar rhoUFlux[3];
                scalar rhoEFlux;

                this->getFlux(&U(c), T(c), p(c), &mesh.normals(index), rhoFlux, rhoUFlux, rhoEFlux);
                viscousFluxUpdate(&U(c), T(c), rhoUFlux, rhoEFlux, index);

                this->operate->div(&rhoFlux, drho, index, false);
                this->operate->div(rhoUFlux, drhoU, index, false);
                this->operate->div(&rhoEFlux, drhoE, index, false);

                scalar aF = sqrt((this->gamma-1)*this->Cp*T(c));
                scalar maxaF = 0;
                for (integer j = 0; j < 3; j++) {
                    maxaF += U(c, j)*mesh.normals(i, j);
                }
                maxaF = fabs(maxaF) + aF;
                this->operate->absDiv(&maxaF, dtc, i, false);
            }
        }
    }
    //faceFluxUpdate(mesh.nLocalFaces, mesh.nFaces, false);
    //cout << "c++: equation 5" << endl;
    //
    
    // CFL computation
    minDtc = 1e100;
    for (integer i = 0; i < mesh.nInternalCells; i++) {
        minDtc = min(2*this->CFL/dtc(i), minDtc);
        drho(i) -= (*this->rhoS)(i);
        for (integer j = 0; j < 3; j++) {
            drhoU(i) -= (*this->rhoUS)(i, j);
        }
        drhoE(i) -= (*this->rhoES)(i);
    }
    //cout << minDtc << endl;
}

template <typename dtype, integer shape1, integer shape2>
void RCF::boundary(const Boundary& boundary, arrType<dtype, shape1, shape2>& phi) {
    const Mesh& mesh = *this->mesh;
    //MPI_Barrier(MPI_COMM_WORLD);

    arrType<dtype, shape1, shape2> phiBuf(mesh.nCells-mesh.nLocalCells);
    integer reqPos;
    if (mesh.nRemotePatches > 0) {
        reqPos = reqIndex/(2*mesh.nRemotePatches);
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
                        phi(c, j, k) = phi(p, j, k);
                    }
                }
            }
        } else if (patchType == "zeroGradient" || patchType == "empty" || patchType == "inletOutlet") {
            for (integer i = 0; i < nFaces; i++) {
                integer p = mesh.owner(startFace + i);
                integer c = cellStartFace + i;
                for (integer j = 0; j < shape1; j++) {
                    for (integer k = 0; k < shape2; k++) {
                        phi(c, j, k) = phi(p, j, k);
                    }
                }
            }
        } else if (patchType == "symmetryPlane" || patchType == "slip") {
            if ((shape1 == 3) && (shape2 == 1)) {
                for (integer i = 0; i < nFaces; i++) {
                    integer f = startFace + i;
                    integer c = cellStartFace + i;
                    integer p = mesh.owner(f);
                    dtype phin = 0.;
                    for (integer j = 0; j < 3; j++) {
                        phin += mesh.normals(f, j)*phi(p, j);
                    }
                    for (integer j = 0; j < 3; j++) {
                        phi(c, j) = phi(p, j) - mesh.normals(f, j)*phin;
                    }
                }
            } else {
                for (integer i = 0; i < nFaces; i++) {
                    integer p = mesh.owner(startFace + i);
                    integer c = cellStartFace + i;
                    for (integer j = 0; j < shape1; j++) {
                        for (integer k = 0; k < shape2; k++) {
                            phi(c, j, k) = phi(p, j, k);
                        }
                    }
                }
            }
        } else if (patchType == "fixedValue") {
            arrType<uscalar, shape1, shape2> phiVal(nFaces, patch.second.at("_value"));

            for (integer i = 0; i < nFaces; i++) {
                integer c = cellStartFace + i;
                for (integer j = 0; j < shape1; j++) {
                    for (integer k = 0; k < shape2; k++) {
                        phi(c, j, k) = phiVal(i, j, k);
                    }
                }
            }
        } else if (patchType == "CBC_UPT") {
            umat Uval(nFaces, patch.second.at("_U0"));
            uvec Tval(nFaces, patch.second.at("_T0"));
            uvec pval(nFaces, patch.second.at("_p0"));
            for (integer i = 0; i < nFaces; i++) {
                integer c = cellStartFace + i;
                for (integer j = 0; j < 3; j++) {
                    (*this->U)(c, j) = Uval(i, j);
                }
                (*this->T)(c) = Tval(i);
                (*this->p)(c) = pval(i);
            }
        } else if (patchType == "CBC_TOTAL_PT") {
            uvec Tt(nFaces, patch.second.at("_Tt"));
            uvec pt(nFaces, patch.second.at("_pt"));
            umat *direction;
            if (patch.second.count("_direction")) {
                direction = new umat(nFaces, patch.second.at("_direction"));
            } else {
                direction = new umat(nFaces, &mesh.normals(startFace));
            }

            for (integer i = 0; i < nFaces; i++) {
                integer c = cellStartFace + i;
                integer o = mesh.owner(startFace + i);
                scalar Un = 0;
                for (integer j = 0; j < 3; j++) {
                    Un = Un + (*this->U)(o, j)*(*direction)(i, j);
                }
                for (integer j = 0; j < 3; j++) {
                    (*this->U)(c, j) = Un*(*direction)(i, j);
                }
                scalar T = Tt(i)-0.5*Un*Un/this->Cp;
                (*this->T)(c) = T;
                (*this->p)(c) = pt(i)*pow(T/Tt(i), this->gamma/(this->gamma-1));
            }
            delete direction;
        } else if (patchType == "processor" || patchType == "processorCyclic") {
            //cout << "hello " << patchID << endl;
            integer bufStartFace = cellStartFace - mesh.nLocalCells;
            integer size = nFaces*shape1*shape2;
            integer dest = stoi(patchInfo.at("neighbProcNo"));
            for (integer i = 0; i < nFaces; i++) {
                integer p = mesh.owner(startFace + i);
                integer b = bufStartFace + i;
                for (integer j = 0; j < shape1; j++) {
                    for (integer k = 0; k < shape2; k++) {
                        phiBuf(b, j, k) = phi(p, j, k);
                    }
                }
            }
            AMPI_Request *req = (AMPI_Request*) this->req;
            integer tag = this->stage*100 + this->reqField*10 + mesh.tags.at(patchID);
            //cout << patchID << " " << tag << endl;
            AMPI_Isend(&phiBuf(bufStartFace), size, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD, &req[this->reqIndex]);
            AMPI_Irecv(&phi(cellStartFace), size, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD, &req[this->reqIndex+1]);
            this->reqIndex += 2;
        }
        else if (patchType == "calculated") {
        } 
        else {
            cout << "patch not found " << patchType << " for " << patchID << endl;
        }
    }
    this->reqField++;
}

void RCF::boundaryInit(integer startField) {
    this->reqIndex = 0;
    this->reqField = startField;
    if (mesh->nRemotePatches > 0) {
        //MPI_Barrier(MPI_COMM_WORLD);
        this->req = (void *)new AMPI_Request[2*3*mesh->nRemotePatches];
    }
}

void RCF::boundaryEnd() {
    if (mesh->nRemotePatches > 0) {
        AMPI_Waitall(2*3*mesh->nRemotePatches, ((AMPI_Request*)this->req), MPI_STATUSES_IGNORE);
        delete[] ((AMPI_Request*)this->req);
        MPI_Barrier(MPI_COMM_WORLD);
    }
}


template void RCF::boundary(const Boundary& boundary, arrType<scalar, 1, 1>& phi);
template void RCF::boundary(const Boundary& boundary, arrType<scalar, 3, 1>& phi);
template void RCF::boundary(const Boundary& boundary, arrType<scalar, 3, 3>& phi);
#ifdef ADIFF
    template void RCF::boundary(const Boundary& boundary, arrType<uscalar, 1, 1>& phi);
    template void RCF::boundary(const Boundary& boundary, arrType<uscalar, 3, 1>& phi);
    template void RCF::boundary(const Boundary& boundary, arrType<uscalar, 3, 3>& phi);
#endif
