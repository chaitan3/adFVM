#define NO_IMPORT_ARRAY
#include "matop.hpp"
#define nrhs 5

#define CHKERRQ(ans) { petscAssert((ans), __FILE__, __LINE__); }
inline void petscAssert(PetscErrorCode code, const char *file, int line, bool abort=true)
{
   if (code != 0) 
   {
      fprintf(stderr,"petscAssert: %d %s %d\n", code, file, line);
      assert(!abort);
      if (abort) exit(code);
   }
}

Matop::Matop() {
    const Mesh& mesh = *meshp;
    integer argc = 0;
    PetscErrorCode error = PetscInitialize(&argc, NULL, NULL, NULL);
    if (error != 0) {
        cout << "petsc error" << endl;
        exit(1);
    }

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

Matop::~Matop () {
    PetscFinalize();
}

int Matop::heat_equation(const arrType<scalar, nrhs>& u, const vec& DT, const scalar dt, arrType<scalar, nrhs>& un) {
    const Mesh& mesh = *meshp;
    Vec x, b;
    Mat A;

    PetscInt n = mesh.nInternalCells;
    PetscInt il, ih;
    PetscInt jl, jh;
    
    CHKERRQ(MatCreate(PETSC_COMM_WORLD, &A));
    CHKERRQ(MatSetSizes(A, n, n, PETSC_DETERMINE, PETSC_DETERMINE));
    MatSetType(A, "aij");
    //MatSetFromOptions(A);
    if (mesh.nProcs > 1) {
        CHKERRQ(MatMPIAIJSetPreallocation(A, 7, NULL, 6, NULL));
    } else {
        CHKERRQ(MatSeqAIJSetPreallocation(A, 7, NULL));
    }
    //MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    CHKERRQ(MatGetOwnershipRange(A, &il, &ih));
    CHKERRQ(MatGetOwnershipRangeColumn(A, &jl, &jh));

    vec faceData(mesh.nFaces);
    for (PetscInt j = 0; j < mesh.nFaces; j++) {
        faceData(j) = mesh.areas(j)*DT(j)/mesh.deltas(j);
    }

    for (PetscInt j = il; j < ih; j++) {
        PetscInt index = j-il;
        scalar neighbourData[6];
        scalar cellData = 0;
        PetscInt cols[6];
        for (PetscInt k = 0; k < 6; k++) {
            PetscInt f = mesh.cellFaces(index, k);
            neighbourData[k] = -faceData(f)/mesh.volumes(index);
            cols[k] = mesh.cellNeighbours(index, k);
            if (cols[k] > -1) {
                cols[k] += jl;
            } 
            if ((cols[k] > -1) || (f >= mesh.nLocalFaces)) {
                cellData -= neighbourData[k];
            }
        }
        CHKERRQ(MatSetValues(A, 1, &j, 6, cols, neighbourData, INSERT_VALUES));
        CHKERRQ(MatSetValue(A, j, index + jl, cellData + 1./dt, INSERT_VALUES));
    }


    //const integer* ranges = new integer[mesh.nProcs+1];
    const PetscInt* ranges;
    CHKERRQ(MatGetOwnershipRangesColumn(A, &ranges));

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
            CHKERRQ(MatSetValue(A, index, neighbourIndex, data, INSERT_VALUES));
        }
    } 
    //delete[] ranges;

    CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY)); CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    
    KSP ksp;
    PC pc;
    CHKERRQ(KSPCreate(PETSC_COMM_WORLD, &(ksp)));
    CHKERRQ(KSPSetOperators(ksp, A, A));
    CHKERRQ(KSPSetType(ksp, KSPGMRES));
    //CHKERRQ(KSPSetType(ksp, KSPPREONLY));
    CHKERRQ(KSPGetPC(ksp, &(pc)));
    //double rtol, atol, dtol;
    //int maxit;
    //KSPGetTolerances(ksp, &rtol, &atol, &dtol, &maxit);
    //cout << rtol << " " << atol << " " << dtol << " " << maxit << endl;
    //KSPSetTolerances(ksp, 1e-4, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
    PCSetType(pc, PCHYPRE);
    //CHKERRQ(PCSetType(pc, PCJACOBI));
    //CHKERRQ(PCSetType(pc, PCLU));
    //CHKERRQ(PCFactorSetMatSolverPackage(pc,MATSOLVERSUPERLU_DIST));


    //KSPSetFromOptions(ksp);
    CHKERRQ(KSPSetUp(ksp));

    CHKERRQ(MatCreateVecs(A, &x, &b));
    //scalar *data1 = new scalar[n];
    //VecPlaceArray(b, data1);
    for (integer i = 0; i < nrhs; i++) {
        scalar *data1, *data2;
        VecGetArray(b, &data1);
        for (integer j = 0; j < n; j++) {
            data1[j] = u(j, i)/dt;
            //VecSetValue(b, j + jl, u(j,i)/dt, INSERT_VALUES);
        }
        VecRestoreArray(b, &data1);
        //VecAssemblyBegin(b);
        //VecAssemblyEnd(b);
        CHKERRQ(KSPSolve(ksp, b, x));
        VecGetArray(x, &data2);
        for (integer j = 0; j < n; j++) {
            un(j, i) = data2[j];
        }
        VecRestoreArray(x, &data2);
    }
    //VecResetArray(b);
    //delete[] data1;

    CHKERRQ(KSPDestroy(&ksp));
    CHKERRQ(VecDestroy(&b));
    CHKERRQ(VecDestroy(&x));
    CHKERRQ(MatDestroy(&A));
    return 0;
}

