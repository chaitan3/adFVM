#ifdef MATOP

#include "matop.hpp"
#include "code.hpp"
#define nrhs 5

Matop::Matop(RCF* rcf) {
    const Mesh& mesh = *meshp;

    for (auto& patch: mesh.boundary) {
        auto& patchInfo = patch.second;
        integer startFace, nFaces;
        tie(startFace, nFaces) = mesh.boundaryFaces.at(patch.first);
        if (startFace >= mesh.nLocalFaces && nFaces > 0) {
            ivec tmp(nFaces, patchInfo.at("loc_neighbourIndices"));
            //cout << patch.first << tmp(0) << " " << tmp(1) << endl;
            boundaryNeighbours[patch.first] = move(tmp);
            boundaryProcs[patch.first] = stoi(patchInfo.at("neighbProcNo"));
        }
    }
}


  

//void Matop::viscosity(const vec& rho, const mat& rhoU, const vec& rhoE, vec& M_2norm, vec& DT, scalar scaling, bool report) {
void Matop::viscosity(const vec& rho, const mat& rhoU, const vec& rhoE, vec& M_2norm, vec& DT, scalar scaling, bool report) {
    const Mesh& mesh = *meshp;


    mat U(mesh.nCells, true);
    vec T(mesh.nCells, true);
    vec p(mesh.nCells, true);

    Function_primitive(mesh.nInternalCells, &rho(0), &rhoU(0), &rhoE(0), &U(0), &T(0), &p(0));
    rcf->boundaryUPT(U, T, p);

    arrType<scalar, 3, 3> gradU(mesh.nInternalCells, true);
    vec divU(mesh.nInternalCells, true);
    arrType<scalar, 1, 3> gradp(mesh.nInternalCells, true);
    arrType<scalar, 1, 3> gradc(mesh.nInternalCells, true);
    //
    #define gradUpdate(i, n, func) \
        func(n, \
                &U(0), &T(0), &p(0), \
                &mesh.areas(i), &mesh.volumesL(i), &mesh.volumesR(i), \
                &mesh.weights(i), &mesh.deltas(i), &mesh.normals(i), \
                &mesh.linearWeights(i), &mesh.quadraticWeights(i), \
                &mesh.owner(i), &mesh.neighbour(i), \
                &gradU(0), &divU(0), &gradp(0), &gradc(0));

    gradUpdate(0, mesh.nInternalFaces, Function_computeGradients);
    rcf->boundaryEnd();    
    for (auto& patch: mesh.boundary) {
        auto& patchInfo = patch.second;
        integer startFace, nFaces;
        tie(startFace, nFaces) = mesh.boundaryFaces.at(patch.first);
        //if (patchInfo.at("type") == "cyclic") {
        if ((patchInfo.at("type") == "cyclic") ||
            (patchInfo.at("type") == "processor") ||
            (patchInfo.at("type") == "processorCyclic")) {
            gradUpdate(startFace, nFaces, Function_coupledComputeGradients);
        } else {
            gradUpdate(startFace, nFaces, Function_boundaryComputeGradients);
        }
    }
    arrType<scalar, 5, 5> MS(mesh.nInternalCells, true);
    Function_viscosity(mesh.nInternalCells, &U(0), &T(0), &p(0), &gradU(0), &divU(0), &gradp(0), &gradc(0), &MS(0));

    getMaxEigenvalue(MS, M_2norm);
    
    //// compute max eigenvalue
    scalar norm = this->norm;
    if (report || norm < 0) {
      scalar V = 0;
      scalar N = 0;
      for (integer i = 0; i < mesh.nInternalCells; i++) {
          N += M_2norm(i)*M_2norm(i)*mesh.volumes(i);
          V += mesh.volumes(i);
      }
      scalar val[2] = {N, V};
      scalar gval[2];
      MPI_Allreduce(&val, &gval, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      norm = sqrt(gval[0]/gval[1]);
    } 
    for (integer i = 0; i < mesh.nInternalCells; i++) {
      M_2norm(i) *= scaling/norm;
    }
    rcf->boundaryInit(0, 1);
    rcf->boundary(mesh.defaultBoundary, M_2norm);
    rcf->boundaryEnd(1);
    for (integer i = 0; i < mesh.nFaces; i++) {
        integer p = mesh.owner(i);
        integer n = mesh.neighbour(i);
        scalar w = mesh.weights(i);
        DT(i) = M_2norm(n)*(1-w) + M_2norm(p)*w;
    }
}

void Matop::heat_equation(RCF *rcf, const arrType<scalar, nrhs>& u, const vec& DT, const scalar dt, arrType<scalar, nrhs>& un) {
    const Mesh& mesh = *meshp;
    Vec x, b;
    Mat A;


    integer n = mesh.nInternalCells;
    integer il, ih;
    integer jl, jh;

    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetSizes(A, n, n, PETSC_DETERMINE, PETSC_DETERMINE);
    MatSetType(A, "aij");
    //MatSetFromOptions(A);
    if (mesh.nProcs > 1) {
        MatMPIAIJSetPreallocation(A, 7, NULL, 6, NULL);
    } else {
        MatSeqAIJSetPreallocation(A, 7, NULL);
    }
    //MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatGetOwnershipRange(A, &il, &ih);
    MatGetOwnershipRangeColumn(A, &jl, &jh);

    vec faceData(mesh.nFaces);
    for (integer j = 0; j < mesh.nFaces; j++) {
        faceData(j) = mesh.areas(j)*DT(j)/mesh.deltas(j);
    }

    for (integer j = il; j < ih; j++) {
        integer index = j-il;
        scalar neighbourData[6];
        scalar cellData = 0;
        integer cols[6];
        for (integer k = 0; k < 6; k++) {
            integer f = mesh.cellFaces(index, k);
            neighbourData[k] = -faceData(f)/mesh.volumes(index);
            cols[k] = mesh.cellNeighbours(index, k);
            if (cols[k] > -1) {
                cols[k] += jl;
            } 
            if ((cols[k] > -1) || (f >= mesh.nLocalFaces)) {
                cellData -= neighbourData[k];
            }
        }
        MatSetValues(A, 1, &j, 6, cols, neighbourData, INSERT_VALUES);
        MatSetValue(A, j, index + jl, cellData + 1./dt, INSERT_VALUES);
    }


    //const integer* ranges = new integer[mesh.nProcs+1];
    const integer* ranges;
    MatGetOwnershipRangesColumn(A, &ranges);

    for (auto& patch: boundaryNeighbours) {
        auto& neighbourIndices = patch.second;
        integer startFace, nFaces;
        tie(startFace, nFaces) = mesh.boundaryFaces.at(patch.first);
        integer proc = boundaryProcs.at(patch.first);//stoi(patchInfo.at("neighbProcNo"));
        for (integer j = 0; j < nFaces; j++) {
            integer f = startFace + j;
            integer p = mesh.owner(f);
            integer index = il + p;
            integer neighbourIndex = ranges[proc] + neighbourIndices(j);
            scalar data = -faceData(f)/mesh.volumes(p);
            MatSetValue(A, index, neighbourIndex, data, INSERT_VALUES);
        }
    } 
    //delete[] ranges;

    MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);

    
    KSP ksp;
    PC pc;
    KSPCreate(PETSC_COMM_WORLD, &(ksp));
    KSPSetOperators(ksp, A, A);
    KSPSetType(ksp, KSPGMRES);
    //KSPSetType(ksp, KSPPREONLY);
    KSPGetPC(ksp, &(pc));
    //double rtol, atol, dtol;
    //int maxit;
    //KSPGetTolerances(ksp, &rtol, &atol, &dtol, &maxit);
    //cout << rtol << " " << atol << " " << dtol << " " << maxit << endl;
    //KSPSetTolerances(ksp, 1e-4, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
    PCSetType(pc, PCHYPRE);
    //PCSetType(pc, PCLU);
    //PCFactorSetMatSolverPackage(pc,MATSOLVERSUPERLU_DIST);


    //KSPSetFromOptions(ksp);
    KSPSetUp(ksp);

    MatCreateVecs(A, &x, &b);
    scalar *data1 = new scalar[n];
    VecPlaceArray(b, data1);
    for (integer i = 0; i < nrhs; i++) {
        for (integer j = 0; j < n; j++) {
            data1[j] = u(j, i)/dt;
        }
        KSPSolve(ksp, b, x);
        scalar *data2;
        VecGetArray(x, &data2);
        for (integer j = 0; j < n; j++) {
            un(j, i) = data2[j];
        }
        VecRestoreArray(x, &data2);
    }
    VecResetArray(b);
    delete[] data1;

    //KSPDestroy(&ksp);
    VecDestroy(&b);
    VecDestroy(&x);
    MatDestroy(&A);
}


#endif
