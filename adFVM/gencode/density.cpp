#include "density.hpp"
#include "code.hpp"

RCF *rcf;

vec *rhos[nStages+1];
mat *rhoUs[nStages+1];
vec *rhoEs[nStages+1];
mat *Us[nStages];
vec *Ts[nStages];
vec *ps[nStages];
arrType<scalar, 3, 3> *gradUs[nStages];
arrType<scalar, 1, 3> *gradTs[nStages];
arrType<scalar, 1, 3> *gradps[nStages];

void RCF::boundaryUPT(mat& U, vec& T, vec& p) {
    const Mesh& mesh = *meshp;
    for (auto& patch: this->boundaries[2]) {
        string patchType = patch.second.at("type");
        string patchID = patch.first;
        integer startFace, nFaces;
        tie(startFace, nFaces) = mesh.boundaryFaces.at(patchID);
        integer cellStartFace = mesh.nInternalCells + startFace - mesh.nInternalFaces;

        if (patchType == "CBC_UPT") {
            mat Uval(nFaces, patch.second.at("_U0"));
            vec Tval(nFaces, patch.second.at("_T0"));
            vec pval(nFaces, patch.second.at("_p0"));
            for (integer i = 0; i < nFaces; i++) {
                integer c = cellStartFace + i;
                for (integer j = 0; j < 3; j++) {
                    U(c, j) = Uval(i, j);
                }
                T(c) = Tval(i);
                p(c) = pval(i);
            }
        } else if (patchType == "CBC_TOTAL_PT") {
            vec Tt(nFaces, patch.second.at("_Tt"));
            vec pt(nFaces, patch.second.at("_pt"));
            mat Uval(nFaces);
            vec Tval(nFaces);
            vec pval(nFaces);
            for (integer i = 0; i < nFaces; i++) {
                integer c = mesh.owner(startFace + i);
                for (integer j = 0; j < 3; j++) {
                    Uval(i, j) = U(c, j);
                }
                Tval(i) = T(c);
                pval(i) = p(c);
            }
            Function_CBC_TOTAL_PT(nFaces, &Uval(0), &Tval(0), &pval(0), &Tt(0), &pt(0), &mesh.normals(startFace), \
                    &U(cellStartFace), &T(cellStartFace), &p(cellStartFace));
        }
    }
    this->boundaryInit(0);    
    this->boundary(this->boundaries[0], U);
    this->boundary(this->boundaries[1], T);
    this->boundary(this->boundaries[2], p);
    //this->boundaryEnd();    
}

void RCF::equation(const vec& rho, const mat& rhoU, const vec& rhoE, vec& drho, mat& drhoU, vec& drhoE, scalar& obj, scalar& minDtc) {
    // make decision between 1 and 3 a template
    // optimal memory layout? combine everything?
    //cout << "c++: equation 1" << endl;
    const Mesh& mesh = *meshp;

    const integer index = this->stage;

    mat U(mesh.nCells, true);
    vec T(mesh.nCells, true);
    vec p(mesh.nCells, true);

    Function_primitive(mesh.nInternalCells, &rho(0), &rhoU(0), &rhoE(0), &U(0), &T(0), &p(0));
    this->boundaryUPT(U, T, p);
    

    U.info();
    T.info();
    p.info();

    arrType<scalar, 3, 3> gradU(mesh.nCells, true);
    arrType<scalar, 1, 3> gradT(mesh.nCells, true);
    arrType<scalar, 1, 3> gradp(mesh.nCells, true);
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
    this->boundaryEnd();    
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
    ////cout << "gradU " << gradU.checkNAN() << endl;
    //
    ////cout << "c++: equation 3" << endl;
    this->boundaryInit(this->reqField);    
    this->boundary(mesh.defaultBoundary, gradU);
    this->boundary(mesh.defaultBoundary, gradT);
    this->boundary(mesh.defaultBoundary, gradp);
    //this->boundaryEnd();

    gradU.info();
    gradT.info();
    gradp.info();

    //
    vec dtc(mesh.nCells);
    drho.zero();
    drhoU.zero();
    drhoE.zero();
    dtc.zero();

    #define fluxUpdate(i, n, func) \
        func(n, \
                &U(0), &T(0), &p(0), \
                &gradU(0), &gradT(0), &gradp(0), \
                &mesh.areas(i), &mesh.volumesL(i), &mesh.volumesR(i), \
                &mesh.weights(i), &mesh.deltas(i), &mesh.normals(i), \
                &mesh.linearWeights(i), &mesh.quadraticWeights(i), \
                &mesh.owner(i), &mesh.neighbour(i), \
                &drho(0), &drhoU(0), &drhoE(0), &dtc(0));

    fluxUpdate(0, mesh.nInternalFaces, Function_flux);
    this->boundaryEnd();    
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
    //drho.info();
    //drhoU.info();
    //drhoE.info();
    //
    if (index == 0) {
        obj = objective(U, T, p);
    }

    minDtc = 1e100;
    for (integer i = 0; i < mesh.nInternalCells; i++) {
        minDtc = min(2*this->CFL/dtc(i), minDtc);
        //drho(i) -= (*this->rhoS)(i);
        //for (integer j = 0; j < 3; j++) {
        //    drhoU(i) -= (*this->rhoUS)(i, j);
        //}
        //drhoE(i) -= (*this->rhoES)(i);
    }

    Us[index] = new mat(move(U));
    Ts[index] = new vec(move(T));
    ps[index] = new vec(move(p));
    gradUs[index] = new arrType<scalar, 3, 3>(move(gradU));
    gradTs[index] = new arrType<scalar, 1, 3>(move(gradT));
    gradps[index] = new arrType<scalar, 1, 3>(move(gradp));
}

void timeIntegrator_init(const vec& rho, const mat& rhoU, const vec& rhoE, vec& rhoN, mat& rhoUN, vec& rhoEN) {
    rhos[0] = new vec(rho.shape, rho.data);
    rhos[nStages] = new vec(rho.shape, rhoN.data);
    rhoUs[0] = new mat(rhoU.shape, rhoU.data);
    rhoUs[nStages] = new mat(rhoU.shape, rhoUN.data);
    rhoEs[0] = new vec(rhoE.shape, rhoE.data);
    rhoEs[nStages] = new vec(rhoE.shape, rhoEN.data);
    for (integer i = 1; i < nStages; i++) {
        rhos[i] = new vec(rho.shape);
        rhoUs[i] = new mat(rho.shape);
        rhoEs[i] = new vec(rho.shape);
    }
}

void timeIntegrator_exit() {
    for (integer i = 0; i < nStages; i++) {
        if (i > 0) {
            delete rhos[i];
            delete rhoUs[i];
            delete rhoEs[i];
        }
        delete Us[i];
        delete Ts[i];
        delete ps[i];
        delete gradUs[i];
        delete gradTs[i];
        delete gradps[i];
    }
}

tuple<scalar, scalar> euler(const vec& rho, const mat& rhoU, const vec& rhoE, vec& rhoN, mat& rhoUN, vec& rhoEN, scalar t, scalar dt) {
    const Mesh& mesh = *meshp;
    
    timeIntegrator_init(rho, rhoU, rhoE, rhoN, rhoUN, rhoEN);
    vec drho(rho.shape);
    mat drhoU(rhoU.shape);
    vec drhoE(rhoE.shape);
    scalar obj, dtc;
    rcf->stage = 0;
    rcf->equation(rho, rhoU, rhoE, drho, drhoU, drhoE, obj, dtc);

    for (integer i = 0; i < mesh.nInternalCells; i++) {
        rhoN(i) = rho(i) - dt*drho(i);
        for (integer j = 0; j < 3; j++) {
            rhoUN(i, j) = rhoU(i, j) - dt*drhoU(i, j);
        }
        rhoEN(i) = rhoE(i) - dt*drhoE(i);
    }
    return make_tuple(obj, dtc);
}



tuple<scalar, scalar> SSPRK(const vec& rho, const mat& rhoU, const vec& rhoE, vec& rhoN, mat& rhoUN, vec& rhoEN, scalar t, scalar dt) {
    const Mesh& mesh = *meshp;


    const integer n = 3;
    scalar alpha[n][n] = {{1,0,0},{3./4, 1./4, 0}, {1./3, 0, 2./3}};
    scalar beta[n][n] = {{1,0,0}, {0,1./4,0},{0,0,2./3}};
    //scalar gamma[n] = {0, 1, 0.5};
    scalar obj[n], dtc[n];

    timeIntegrator_init(rho, rhoU, rhoE, rhoN, rhoUN, rhoEN);
    vec drho(rho.shape);
    mat drhoU(rhoU.shape);
    vec drhoE(rhoE.shape);

    for (integer stage = 0; stage < n; stage++) {
        //solver.t = solver.t0 + gamma[i]*solver.dt
        rcf->stage = stage;
        //(*rhos[stage]).info();
        //(*rhoUs[stage]).info();
        //(*rhoEs[stage]).info();
        rcf->equation(*rhos[stage], *rhoUs[stage], *rhoEs[stage], drho, drhoU, drhoE, obj[stage], dtc[stage]);
        integer curr = stage + 1;
        scalar b = beta[stage][stage];
        for (integer i = 0; i < mesh.nInternalCells; i++) {
            (*rhos[curr])(i) = -b*drho(i)*dt;
            for (integer j = 0; j < 3; j++) {
                (*rhoUs[curr])(i, j) = -b*drhoU(i, j)*dt;
            }
            (*rhoEs[curr])(i) = -b*drhoE(i)*dt;
        }
        for (integer prev = 0; prev < curr; prev++) {
            scalar a = alpha[stage][prev];
            for (integer i = 0; i < mesh.nInternalCells; i++) {
                (*rhos[curr])(i) += a*(*rhos[prev])(i);
                for (integer j = 0; j < 3; j++) {
                    (*rhoUs[curr])(i, j) += a*(*rhoUs[prev])(i, j);
                }
                (*rhoEs[curr])(i) += a*(*rhoEs[prev])(i);
            }
        }
        //(*rhos[stage]).info();
        //(*rhos[curr]).info();
    }
    return make_tuple(obj[0], dtc[0]);
}


template <typename dtype, integer shape1, integer shape2>
void RCF::boundary(const Boundary& boundary, arrType<dtype, shape1, shape2>& phi) {
    const Mesh& mesh = *meshp;
    //MPI_Barrier(MPI_COMM_WORLD);

    dtype* phiBuf = NULL;
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
            cout << "implement this elsewhere" << endl;
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
            arrType<scalar, shape1, shape2> phiVal(nFaces, patch.second.at("_value"));

            for (integer i = 0; i < nFaces; i++) {
                integer c = cellStartFace + i;
                for (integer j = 0; j < shape1; j++) {
                    for (integer k = 0; k < shape2; k++) {
                        phi(c, j, k) = phiVal(i, j, k);
                    }
                }
            }
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
                        phiBuf[b*shape1*shape2 + j*shape2 + k] = phi(p, j, k);
                    }
                }
            }
            MPI_Request *req = (MPI_Request*) this->req;
            integer tag = (this->stage*1000+1) + this->reqField*100 + mesh.tags.at(patchID);
            //cout << patchID << " " << tag << endl;
            MPI_Isend(&phiBuf[bufStartFace*shape1*shape2], size, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD, &req[this->reqIndex]);
            MPI_Irecv(&phi(cellStartFace), size, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD, &req[this->reqIndex+1]);
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

void RCF::boundaryInit(integer startField, integer nFields) {
    const Mesh& mesh = *meshp;
    this->reqIndex = 0;
    this->reqField = startField;
    if (mesh.nRemotePatches > 0) {
        //MPI_Barrier(MPI_COMM_WORLD);
        this->req = (void *)new MPI_Request[2*nFields*mesh.nRemotePatches];
    }
}

void RCF::boundaryEnd(integer nFields) {
    const Mesh& mesh = *meshp;
    if (mesh.nRemotePatches > 0) {
        MPI_Waitall(2*nFields*mesh.nRemotePatches, ((MPI_Request*)this->req), MPI_STATUSES_IGNORE);
        delete[] ((MPI_Request*)this->req);
        //MPI_Barrier(MPI_COMM_WORLD);
        for (integer i = 0; i < nFields; i++) {
            delete[] this->reqBuf[i];
        }
    }
}

