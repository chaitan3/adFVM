#define NO_IMPORT_ARRAY
#include "matop_petsc.hpp"

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

/*int Matop::heat_equation(vector<ext_vec*> w, vector<ext_vec*> u, const ext_vec& DTF, const ext_vec& dt_vec, vector<ext_vec*> un, bool density) {*/
    //const Mesh& mesh = *meshp;
    //Vec x, b;
    //Mat A;
    //long long start = current_timestamp();
    ////if (mesh.rank == 0) cout << "here1" << endl;
    ////MPI_Barrier(MPI_COMM_WORLD);
    ////exit(1);

    //PetscInt n = mesh.nInternalCells;
    //PetscInt il, ih;
    //PetscInt jl, jh;

    //#ifdef GPU
        //const scalar* faceData = DTF.toHost();
        //const scalar* dt_data = dt_vec.toHost();
        //const scalar dt = dt_data[0];
        //#define VecCreateMPITypeWithArray VecCreateMPICUDAWithArray
        //#define VecCreateTypeWithArray VecCreateSeqCUDAWithArray
        //#define VecPlaceType VecCUDAPlaceArray
        //#define VecResetType VecCUDAResetArray
        //#define SparseType "aijcusparse"
        ////#define SparseType "aijcusp"
    //#else
        //const scalar* faceData = DTF.data;
        //const scalar dt = dt_vec(0);
        //#define VecCreateMPITypeWithArray VecCreateMPIWithArray
        //#define VecCreateTypeWithArray VecCreateSeqWithArray
        //#define VecPlaceType VecPlaceArray
        //#define VecResetType VecResetArray
        //#define SparseType "aij"
    //#endif

    //// assembly time less than 10% on CPU
    //CHKERRQ(MatCreate(PETSC_COMM_WORLD, &A));
    //CHKERRQ(MatSetSizes(A, n, n, PETSC_DETERMINE, PETSC_DETERMINE));
    //MatSetType(A, SparseType);
    ////MatSetFromOptions(A);
    //if (mesh.nProcs > 1) {
        //CHKERRQ(MatMPIAIJSetPreallocation(A, 7, NULL, 6, NULL));
    //} else {
        //CHKERRQ(MatSeqAIJSetPreallocation(A, 7, NULL));
    //}
    ////MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    //CHKERRQ(MatGetOwnershipRange(A, &il, &ih));
    //CHKERRQ(MatGetOwnershipRangeColumn(A, &jl, &jh));
    //long long start4 = current_timestamp();
    //if (mesh.rank == 0) cout << "matop init: " << start4-start << endl;

    //for (PetscInt j = il; j < ih; j++) {
        //PetscInt index = j-il;
        //scalar neighbourData[6];
        //scalar cellData = 0;
        //PetscInt cols[6];
        //for (PetscInt k = 0; k < 6; k++) {
            //PetscInt f = mesh.cellFaces(index, k);
            ////neighbourData[k] = -faceData[f]/mesh.volumes(index);
            //neighbourData[k] = -faceData[f]/mesh.volumes(index)*dt;
            //cols[k] = mesh.cellNeighbours(index, k);
            //if (cols[k] > -1) {
                //cols[k] += jl;
            //} 
            //if ((cols[k] > -1) || (f >= mesh.nLocalFaces)) {
                //cellData -= neighbourData[k];
            //}
            //assert(std::isfinite(faceData[f]));
            //assert(std::isfinite(neighbourData[k]));
        //}
        //assert(std::isfinite(cellData));
        //CHKERRQ(MatSetValues(A, 1, &j, 6, cols, neighbourData, INSERT_VALUES));
        ////CHKERRQ(MatSetValue(A, j, index + jl, cellData + 1./dt, INSERT_VALUES));
        //CHKERRQ(MatSetValue(A, j, index + jl, cellData + 1., INSERT_VALUES));
    //}


    ////const integer* ranges = new integer[mesh.nProcs+1];
    //const PetscInt* ranges;
    //CHKERRQ(MatGetOwnershipRangesColumn(A, &ranges));

    //for (auto& patch: boundaryNeighbours) {
        //auto& neighbourIndices = patch.second;
        //integer startFace, nFaces;
        //tie(startFace, nFaces) = mesh.boundaryFaces.at(patch.first);
        //integer proc = boundaryProcs.at(patch.first);//stoi(patchInfo.at("neighbProcNo"));
        //for (integer j = 0; j < nFaces; j++) {
            //integer f = startFace + j;
            //integer p = mesh.owner(f);
            //integer index = il + p;
            //integer neighbourIndex = ranges[proc] + neighbourIndices(j);
            ////scalar data = -faceData[f]/mesh.volumes(p);
            //scalar data = -faceData[f]/mesh.volumes(p)*dt;
            //CHKERRQ(MatSetValue(A, index, neighbourIndex, data, INSERT_VALUES));
        //}
    //} 

    //CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY)); CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    //long long start2 = current_timestamp();
    //if (mesh.rank == 0) cout << "matop_assembly: " << start2-start << endl;
        
    
    //KSP ksp;
    //PC pc;
    //CHKERRQ(KSPCreate(PETSC_COMM_WORLD, &(ksp)));
    //CHKERRQ(KSPSetOperators(ksp, A, A));
    ////BEST CPU
    //CHKERRQ(KSPSetType(ksp, KSPGMRES));
    ////CHKERRQ(KSPSetType(ksp, KSPGCR));
    ////CHKERRQ(KSPSetType(ksp, KSPBCGS));
    ////BEST GPU
    ////CHKERRQ(KSPSetType(ksp, KSPTFQMR));
    ////CHKERRQ(KSPSetType(ksp, KSPPREONLY));

    ////double rtol, atol, dtol;
    ////int maxit;
    ////KSPGetTolerances(ksp, &rtol, &atol, &dtol, &maxit);
    ////cout << rtol << " " << atol << " " << dtol << " " << maxit << endl;
    ////KSPSetTolerances(ksp, 1e-4, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
    //KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);

    //CHKERRQ(KSPGetPC(ksp, &(pc)));
    ////BEST GPU
    ////CHKERRQ(PCSetType(pc, PCJACOBI));
    ////CHKERRQ(PCSetType(pc, PCASM));
    ////CHKERRQ(PCSetType(pc, PCMG));
    ////CHKERRQ(PCSetType(pc, PCGAMG));
    ////BEST CPU
    //CHKERRQ(PCSetType(pc, PCHYPRE));

    ////CHKERRQ(PCSetType(pc, PCLU));
    ////CHKERRQ(PCFactorSetMatSolverPackage(pc,MATSOLVERSUPERLU_DIST));


    ////KSPSetFromOptions(ksp);
    //CHKERRQ(KSPSetUp(ksp));

    ////scalar *data1 = new scalar[n];
    ////VecPlaceArray(b, data1);
    //if (mesh.nProcs > 1) {
        //CHKERRQ(VecCreateMPITypeWithArray(PETSC_COMM_WORLD, 1, n, PETSC_DECIDE, NULL, &b));
        //CHKERRQ(VecCreateMPITypeWithArray(PETSC_COMM_WORLD, 1, n, PETSC_DECIDE, NULL, &x));
    //} else {
        //CHKERRQ(VecCreateTypeWithArray(PETSC_COMM_WORLD, 1, n, NULL, &b));
        //CHKERRQ(VecCreateTypeWithArray(PETSC_COMM_WORLD, 1, n, NULL, &x));
        ////CHKERRQ(MatCreateVecs(A, &x, &b));
    //}
    //integer i = 0;
    //if (!density) {
        //un[i]->copy(0, u[i]->data, n);
        //i = 1;
    //}
    //for (; i < nrhs; i++) {
        //un[i]->copy(0, u[i]->data, n);
        //CHKERRQ(VecPlaceType(b, u[i]->data));
        //CHKERRQ(VecPlaceType(x, un[i]->data));
        
        //CHKERRQ(KSPSolve(ksp, b, x));
        //KSPConvergedReason reason;
        //KSPGetConvergedReason(ksp, &reason);
        //PetscInt its;
        //KSPGetIterationNumber(ksp, &its);
	//if (mesh.rank == 0) {
	    //PetscPrintf(PETSC_COMM_WORLD,"KSPConvergedReason: %D\n", reason);
	    //PetscPrintf(PETSC_COMM_WORLD,"KSPIterations: %d\n", its);
        //}
        //if (reason < 0)  {
            //return 1;
        //}

        //CHKERRQ(VecResetType(b));
        //CHKERRQ(VecResetType(x));
        //long long start3 = current_timestamp();
        //if (mesh.rank == 0) cout << "ksp: " << start3-start << endl;
    //}
    ////VecResetArray(b);
    ////delete[] data1;
    //#ifdef GPU
        //delete[] dt_data;
        //delete[] faceData;
    //#endif

    //CHKERRQ(KSPDestroy(&ksp));
    //CHKERRQ(MatDestroy(&A));
    //CHKERRQ(VecDestroy(&b));
    //CHKERRQ(VecDestroy(&x));
    //return 0;
/*}*/


int Matop::heat_equation(vector<ext_vec*> w, vector<ext_vec*> u, const ext_vec& DTF, const ext_vec& dt_vec, vector<ext_vec*> un, bool density) {
    const Mesh& mesh = *meshp;
    Vec x, b;
    Mat A;
    long long start = current_timestamp();
    //if (mesh.rank == 0) cout << "here1" << endl;
    //MPI_Barrier(MPI_COMM_WORLD);
    //exit(1);
    //
    //
    //
    

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
    const scalar* rho = w[0]->data;
    const scalar* rhoU = w[1]->data;
    const scalar* rhoE = w[2]->data;

    // assembly time less than 10% on CPU
    CHKERRQ(MatCreate(PETSC_COMM_WORLD, &A));
    CHKERRQ(MatSetSizes(A, nrhs*n, nrhs*n, PETSC_DETERMINE, PETSC_DETERMINE));
    MatSetType(A, SparseType);
    //MatSetFromOptions(A);
    if (mesh.nProcs > 1) {
        CHKERRQ(MatMPIAIJSetPreallocation(A, 7*nrhs, NULL, 6*nrhs, NULL));
    } else {
        CHKERRQ(MatSeqAIJSetPreallocation(A, 7*nrhs, NULL));
    }
    //MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    CHKERRQ(MatGetOwnershipRange(A, &il, &ih));
    CHKERRQ(MatGetOwnershipRangeColumn(A, &jl, &jh));

    CHKERRQ(MatCreateVecs(A, &x, &b));
    long long start4 = current_timestamp();
    if (mesh.rank == 0) cout << "matop init: " << start4-start << endl;

    for (PetscInt j = il; j < ih; j += nrhs) {

        scalar g = 1.4;
        PetscInt index = (j-il)/nrhs;
        scalar u1 = rhoU[index*3]/rho[index];
        scalar u2 = rhoU[index*3+1]/rho[index];
        scalar u3 = rhoU[index*3+2]/rho[index];
        scalar q2 = u1*u1+u2*u2+u3*u3;
        scalar p = (rhoE[index]-rho[index]*q2/2)*(g-1);
        scalar H = g*p/(rho[index]*(g-1)) + q2/2;

        scalar A0[nrhs*nrhs];
        A0[0*nrhs+0] = rho[index];
        A0[0*nrhs+1] = rho[index]*u1;
        A0[0*nrhs+2] = rho[index]*u2;
        A0[0*nrhs+3] = rho[index]*u3;
        A0[0*nrhs+4] = rhoE[index];
        A0[1*nrhs+1] = rho[index]*u1*u1+p;
        A0[1*nrhs+2] = rho[index]*u1*u2;
        A0[1*nrhs+3] = rho[index]*u1*u3;
        A0[1*nrhs+4] = rho[index]*H*u1;
        A0[2*nrhs+2] = rho[index]*u2*u2+p;
        A0[2*nrhs+3] = rho[index]*u2*u3;
        A0[2*nrhs+4] = rho[index]*H*u2;
        A0[3*nrhs+3] = rho[index]*u3*u3+p;
        A0[3*nrhs+4] = rho[index]*H*u3;
        A0[4*nrhs+4] = rho[index]*H*H-g*p*p/(rho[index]*(g-1));
        A0[1*nrhs+0] = A0[0*nrhs+1];
        A0[2*nrhs+1] = A0[1*nrhs+2];
        A0[2*nrhs+0] = A0[0*nrhs+2];
        A0[3*nrhs+2] = A0[2*nrhs+3];
        A0[3*nrhs+1] = A0[1*nrhs+3];
        A0[3*nrhs+0] = A0[0*nrhs+3];
        A0[4*nrhs+3] = A0[3*nrhs+4];
        A0[4*nrhs+2] = A0[2*nrhs+4];
        A0[4*nrhs+1] = A0[1*nrhs+4];
        A0[4*nrhs+0] = A0[0*nrhs+4];
        scalar neighbourData[6];
        scalar cellData = 0;
        PetscInt cols[6];
        for (PetscInt k = 0; k < 6; k++) {
            PetscInt f = mesh.cellFaces(index, k);
            //neighbourData[k] = -faceData[f]/mesh.volumes(index);
            neighbourData[k] = -faceData[f]/mesh.volumes(index)*dt;
            cols[k] = mesh.cellNeighbours(index, k)*nrhs;
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
        scalar bs[nrhs] = {0,0,0,0,0};
        PetscInt indices[nrhs] = {j,j+1,j+2,j+3,j+4};
        for (PetscInt row = 0; row < nrhs; row++) {
            
            for (PetscInt col = 0; col < nrhs; col++) {
                CHKERRQ(MatSetValue(A, row + j, col + j, A0[row*nrhs + col], INSERT_VALUES));
                bs[row] += A0[row*nrhs+col]*u[col]->data[index];
            }
            CHKERRQ(MatSetValue(A, row + j, row + j, cellData, ADD_VALUES));
            PetscInt rowj = row + j;
            CHKERRQ(MatSetValues(A, 1, &rowj, 6, cols, neighbourData, INSERT_VALUES));
            for (PetscInt k = 0; k < 6; k++) {
                if (cols[k] > -1) {
                    cols[k] += 1;
                }
            }
        }
        CHKERRQ(VecSetValues(b, nrhs, indices, bs, INSERT_VALUES));
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
            integer index = il + p*nrhs;
            integer neighbourIndex = ranges[proc] + neighbourIndices(j)*nrhs;
            //scalar data = -faceData[f]/mesh.volumes(p);
            scalar data = -faceData[f]/mesh.volumes(p)*dt;
            for (int row = 0; row <  nrhs; row ++) {
                CHKERRQ(MatSetValue(A, index + row, neighbourIndex + row, data, INSERT_VALUES));
            }
        }
    } 

    CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY)); CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(VecAssemblyBegin(b)); CHKERRQ(VecAssemblyEnd(b));
    long long start2 = current_timestamp();
    if (mesh.rank == 0) cout << "matop_assembly: " << start2-start << endl;
        
    
    KSP ksp;
    PC pc;
    CHKERRQ(KSPCreate(PETSC_COMM_WORLD, &(ksp)));
    CHKERRQ(KSPSetOperators(ksp, A, A));
    //BEST CPU
    CHKERRQ(KSPSetType(ksp, KSPGMRES));
    //CHKERRQ(KSPSetType(ksp, KSPGCR));
    //CHKERRQ(KSPSetType(ksp, KSPBCGS));
    //BEST GPU
    //CHKERRQ(KSPSetType(ksp, KSPTFQMR));
    //CHKERRQ(KSPSetType(ksp, KSPPREONLY));

    //double rtol, atol, dtol;
    //int maxit;
    //KSPGetTolerances(ksp, &rtol, &atol, &dtol, &maxit);
    //cout << rtol << " " << atol << " " << dtol << " " << maxit << endl;
    //KSPSetTolerances(ksp, 1e-4, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
    KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);

    CHKERRQ(KSPGetPC(ksp, &(pc)));
    //BEST GPU
    //CHKERRQ(PCSetType(pc, PCJACOBI));
    //CHKERRQ(PCSetType(pc, PCASM));
    //CHKERRQ(PCSetType(pc, PCMG));
    //CHKERRQ(PCSetType(pc, PCGAMG));
    //BEST CPU
    CHKERRQ(PCSetType(pc, PCHYPRE));

    //CHKERRQ(PCSetType(pc, PCLU));
    //CHKERRQ(PCFactorSetMatSolverPackage(pc,MATSOLVERSUPERLU_DIST));


    //KSPSetFromOptions(ksp);
    CHKERRQ(KSPSetUp(ksp));

    //scalar *data1 = new scalar[n];
    //VecPlaceArray(b, data1);
    
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

    scalar *res;
    CHKERRQ(VecGetArray(x, &res));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < nrhs; j++) {
            un[j]->data[i] = res[i*nrhs + j];
        }
    }

    long long start3 = current_timestamp();
    if (mesh.rank == 0) cout << "ksp: " << start3-start << endl;
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
