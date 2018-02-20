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

int Matop::heat_equation(vector<ext_vec*> u, const ext_vec& DTF, const ext_vec& dt_vec, vector<ext_vec*> un) {
    const Mesh& mesh = *meshp;
    Vec x, b;
    Mat A;
    long long start = current_timestamp();
    //if (mesh.rank == 0) cout << "here1" << endl;
    //MPI_Barrier(MPI_COMM_WORLD);
    //exit(1);

    PetscInt n = mesh.nInternalCells;
    PetscInt il, ih;
    PetscInt jl, jh;

    #ifdef GPU
        const scalar* faceData = DTF.toHost();
        const scalar* dt_data = dt_vec.toHost();
        const scalar dt = dt_data[0];
        #define VecCreateMPITypeWithArray VecCreateMPICUDAWithArray
        #define VecCreateTypeWithArray VecCreateSeqCUDAWithArray
        #define VecPlaceType VecCUDAPlaceArray
        #define VecResetType VecCUDAResetArray
        #define SparseType "aijcusparse"
        //#define SparseType "aijcusp"
    #else
        const scalar* faceData = DTF.data;
        const scalar dt = dt_vec(0);
        #define VecCreateMPITypeWithArray VecCreateMPIWithArray
        #define VecCreateTypeWithArray VecCreateSeqWithArray
        #define VecPlaceType VecPlaceArray
        #define VecResetType VecResetArray
        #define SparseType "aij"
    #endif

    // assembly time less than 10% on CPU
    CHKERRQ(MatCreate(PETSC_COMM_WORLD, &A));
    CHKERRQ(MatSetSizes(A, n, n, PETSC_DETERMINE, PETSC_DETERMINE));
    MatSetType(A, SparseType);
    //MatSetFromOptions(A);
    if (mesh.nProcs > 1) {
        CHKERRQ(MatMPIAIJSetPreallocation(A, 7, NULL, 6, NULL));
    } else {
        CHKERRQ(MatSeqAIJSetPreallocation(A, 7, NULL));
    }
    //MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    CHKERRQ(MatGetOwnershipRange(A, &il, &ih));
    CHKERRQ(MatGetOwnershipRangeColumn(A, &jl, &jh));
    long long start4 = current_timestamp();
    if (mesh.rank == 0) cout << start4-start << endl;

    for (PetscInt j = il; j < ih; j++) {
        PetscInt index = j-il;
        scalar neighbourData[6];
        scalar cellData = 0;
        PetscInt cols[6];
        for (PetscInt k = 0; k < 6; k++) {
            PetscInt f = mesh.cellFaces(index, k);
            //neighbourData[k] = -faceData[f]/mesh.volumes(index);
            neighbourData[k] = -faceData[f]/mesh.volumes(index)*dt;
            cols[k] = mesh.cellNeighbours(index, k);
            if (cols[k] > -1) {
                cols[k] += jl;
            } 
            if ((cols[k] > -1) || (f >= mesh.nLocalFaces)) {
                cellData -= neighbourData[k];
            }
            assert(std::isfinite(faceData[f]));
            assert(std::isfinite(neighbourData[k]));
        }
        assert(std::isfinite(cellData));
        CHKERRQ(MatSetValues(A, 1, &j, 6, cols, neighbourData, INSERT_VALUES));
        //CHKERRQ(MatSetValue(A, j, index + jl, cellData + 1./dt, INSERT_VALUES));
        CHKERRQ(MatSetValue(A, j, index + jl, cellData + 1., INSERT_VALUES));
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
            //scalar data = -faceData[f]/mesh.volumes(p);
            scalar data = -faceData[f]/mesh.volumes(p)*dt;
            CHKERRQ(MatSetValue(A, index, neighbourIndex, data, INSERT_VALUES));
        }
    } 

    CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY)); CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    long long start2 = current_timestamp();
    if (mesh.rank == 0) cout << start2-start << endl;
        
    
    KSP ksp;
    PC pc;
    CHKERRQ(KSPCreate(PETSC_COMM_WORLD, &(ksp)));
    CHKERRQ(KSPSetOperators(ksp, A, A));
    CHKERRQ(KSPSetType(ksp, KSPGMRES));
    //CHKERRQ(KSPSetType(ksp, KSPGCR));
    //CHKERRQ(KSPSetType(ksp, KSPBCGS));
    //CHKERRQ(KSPSetType(ksp, KSPTFQMR));
    //CHKERRQ(KSPSetType(ksp, KSPPREONLY));

    //double rtol, atol, dtol;
    //int maxit;
    //KSPGetTolerances(ksp, &rtol, &atol, &dtol, &maxit);
    //cout << rtol << " " << atol << " " << dtol << " " << maxit << endl;
    //KSPSetTolerances(ksp, 1e-4, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
    KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);

    CHKERRQ(KSPGetPC(ksp, &(pc)));
    //CHKERRQ(PCSetType(pc, PCJACOBI));
    //CHKERRQ(PCSetType(pc, PCASM));
    //CHKERRQ(PCSetType(pc, PCMG));
    CHKERRQ(PCSetType(pc, PCGAMG));
    //CHKERRQ(PCSetType(pc, PCHYPRE));

    //CHKERRQ(PCSetType(pc, PCLU));
    //CHKERRQ(PCFactorSetMatSolverPackage(pc,MATSOLVERSUPERLU_DIST));


    //KSPSetFromOptions(ksp);
    CHKERRQ(KSPSetUp(ksp));

    //scalar *data1 = new scalar[n];
    //VecPlaceArray(b, data1);
    if (mesh.nProcs > 1) {
        CHKERRQ(VecCreateMPITypeWithArray(PETSC_COMM_WORLD, 1, n, PETSC_DECIDE, NULL, &b));
        CHKERRQ(VecCreateMPITypeWithArray(PETSC_COMM_WORLD, 1, n, PETSC_DECIDE, NULL, &x));
    } else {
        CHKERRQ(VecCreateTypeWithArray(PETSC_COMM_WORLD, 1, n, NULL, &b));
        CHKERRQ(VecCreateTypeWithArray(PETSC_COMM_WORLD, 1, n, NULL, &x));
        //CHKERRQ(MatCreateVecs(A, &x, &b));
    }
    for (integer i = 0; i < nrhs; i++) {
        un[i]->copy(0, u[i]->data, n);
        CHKERRQ(VecPlaceType(b, u[i]->data));
        CHKERRQ(VecPlaceType(x, un[i]->data));
        
        CHKERRQ(KSPSolve(ksp, b, x));
        KSPConvergedReason reason;
        KSPGetConvergedReason(ksp, &reason);
        PetscInt its;
        KSPGetIterationNumber(ksp, &its);
	if (mesh.rank == 0) {
	    PetscPrintf(PETSC_COMM_WORLD,"KSPConvergedReason: %D\n", reason);
	    PetscPrintf(PETSC_COMM_WORLD,"KSPIterations: %d\n", its);
        }
        if (reason < 0)  {
            return 1;
        }

        CHKERRQ(VecResetType(b));
        CHKERRQ(VecResetType(x));
        long long start3 = current_timestamp();
        if (mesh.rank == 0) cout << start3-start << endl;
    }
    //VecResetArray(b);
    //delete[] data1;
    #ifdef GPU
        delete[] dt_data;
        delete[] faceData;
    #endif

    CHKERRQ(KSPDestroy(&ksp));
    CHKERRQ(MatDestroy(&A));
    CHKERRQ(VecDestroy(&b));
    CHKERRQ(VecDestroy(&x));
    return 0;
}



