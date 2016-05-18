from field import Field, IOField
import parallel
from parallel import pprint

import numpy as np
import scipy.sparse as sp
import time

import petsc4py
petsc4py.init()
from petsc4py import PETSc

class Matrix(object):
    def __init__(self, A, b):
        self.A = A
        self.b = b

    @classmethod
    def create(self, m, n, nnz=(2,1), nrhs=1):
        A = PETSc.Mat()
        A.create(PETSc.COMM_WORLD)
        A.setSizes(((m, PETSc.DECIDE), (n, PETSc.DECIDE)))
        A.setType('aij')
        A.setPreallocationNNZ(nnz) 

        b = PETSc.Mat()
        b.create(PETSc.COMM_WORLD)
        b.setSizes(((m, PETSc.DECIDE), (PETSc.DECIDE, nrhs)))
        b.setType('dense')
        b.setUp()

        #b = A.createVecLeft()
        #b.set(0)
        return self(A, b)

    def __add__(self, b):
        if isinstance(b, Matrix):
            return self.__class__(self.A + b.A, self.b + b.b)
        else:
            raise Exception("WTF")

    def __sub__(self, b):
        return self.__add__(-b)

    def __neg__(self):
        return self.__class__(-self.A, -self.b)
    
    def __rsub__(self, b):
        raise Exception("WTF")

    def __radd__(self, b):
        return self.__add__(self, b)

    def __mul__(self, b):
        return self.__class__(self.A * b, self.b * b)

    def __rmul__(self, b):
        return self.__class__(b * self.A, b * self.b)

    def solve(self):
        start = time.time()
        ksp = PETSc.KSP()
        ksp.create(PETSc.COMM_WORLD)
        #ksp.setType('preonly')
        #pc = ksp.getPC()
        #pc.setType('lu')
        #pc.setFactorSolverPackage('superlu')
        ksp.setType('gmres')
        ksp.getPC().setType('jacobi')
        x = self.A.createVecRight()
        X = []

        ksp.setOperators(self.A)
        ksp.setFromOptions()
        for i in range(0, self.b.getSize()[1]):
            x.set(0)
            b = self.b.getColumnVector(i)
            ksp.solve(-b, x)
            X.append(x.getArray().copy().reshape(-1,1))
        end = time.time()
        pprint('Time to solve linear system:', end-start)
        return np.hstack(X)

# cyclic patches support, and better BC support
def laplacian(phi, DT):
    dim = phi.dimensions
    mesh = phi.mesh.origMesh
    #n = mesh.nLocalCells
    #m = mesh.nFaces - (mesh.nCells - mesh.nLocalCells)
    l = mesh.nFaces
    m = mesh.nInternalFaces
    n = mesh.nInternalCells
    o = mesh.nFaces - (mesh.nCells - mesh.nLocalCells)
    nrhs = phi.dimensions[0]

    snGradM = Matrix.create(l, n, 2, nrhs)
    snGradOp, snGradb = snGradM.A, snGradM.b

    il, ih = snGradOp.getOwnershipRange()
    jl, jh = snGradOp.getOwnershipRangeColumn()
    #data = (mesh.areas/mesh.deltas).flatten()
    data = (mesh.areas*DT.field/mesh.deltas).flatten()
    row = np.arange(0, l, dtype=np.int32)
    data = np.concatenate((-data, data[:m], data[o:]))
    row = np.concatenate((row, row[:m], row[o:]))
    procCols = mesh.neighbour[o:].copy()
    procRanges = snGradOp.getOwnershipRangesColumn()
    for patchID in phi.mesh.remotePatches:
        patch = mesh.boundary[patchID]
        startFace = patch['startFace']-o
        endFace = startFace + patch['nFaces']
        proc = patch['neighbProcNo']
        procCols[startFace:endFace] = patch['neighbourIndices']
        procCols[startFace:endFace] += -jl + procRanges[proc]
    col = np.concatenate((mesh.owner, mesh.neighbour[:m], procCols))
    snGradOp.setValuesRCV(il + row.reshape(-1,1), jl + col.reshape(-1,1), data.reshape(-1,1))
    snGradOp.assemble()

    indices = np.arange(m, o).astype(np.int32)
    data = data[m:o].reshape(-1,1)*phi.field[mesh.neighbour[m:o]]
    cols = np.arange(0, nrhs).astype(np.int32)
    snGradb.setValues(il + indices, cols, data)
    snGradb.assemble()
    
    sumOp = Matrix.create(n, l, 6).A
    il, ih = sumOp.getOwnershipRange()
    jl, jh = sumOp.getOwnershipRangeColumn()
    indices = mesh.sumOp.indices
    indptr = mesh.sumOp.indptr
    data = mesh.sumOp.data
    sumOp.setValuesIJV(indptr, jl + indices, data)
    sumOp.assemble()
    M = snGradM.__rmul__(sumOp)

    volOp = Matrix.create(n, n, 1).A
    diag = volOp.createVecRight()
    il, ih = diag.getOwnershipRange()
    indices = np.arange(0, n).astype(np.int32)
    data = 1./mesh.volumes.flatten()
    diag.setValues(il + indices, data)
    diag.assemble()
    volOp.setDiagonal(diag)
    volOp.assemble()
    M = M.__rmul__(volOp)

    return M

def ddt(phi, dt):
    mesh = phi.mesh.origMesh
    n = mesh.nInternalCells
    nrhs = phi.dimensions[0]
    M = Matrix.create(n, n, 1, nrhs)
    A, b = M.A, M.b

    il, ih = A.getOwnershipRange()
    diag = np.arange(0,n).astype(np.int32).reshape(-1,1)
    v = np.ones_like(diag)/dt
    A.setValuesRCV(il + diag, il + diag, v)
    A.assemble()

    oldPhi = phi.old[:n]
    cols = np.arange(0, nrhs).astype(np.int32)
    b.setValues(il + diag[:,0], cols, -oldPhi/dt)
    b.assemble()

    return M

def BCs(phi, M):
    mesh = phi.mesh.origMesh
    m = mesh.nLocalCells-mesh.nInternalCells
    n = mesh.nLocalCells
    data = np.concatenate((np.ones(m),-np.ones(m)))
    row = np.arange(0, m)
    row = np.concatenate((row, row))
    col = np.zeros(2*m)
    for patchID in phi.mesh.origPatches:
        patch = mesh.boundary[patchID]
        startFace = patch['startFace']
        endFace = startFace + patch['nFaces']
        cellStartFace = startFace-mesh.nInternalFaces
        cellEndFace = cellStartFace + patch['nFaces']
        if patch['type'] == 'cyclic':
            neighbourPatch = mesh.boundary[patch['neighbourPatch']]   
            neighbourStartFace = neighbourPatch['startFace']
            neighbourEndFace = neighbourStartFace + patch['nFaces']
            owner = mesh.owner[neighbourStartFace:neighbourEndFace] 
        else:
            owner = mesh.owner[startFace:endFace]
        col[cellStartFace:cellEndFace] = owner
        col[m + cellStartFace:m + cellEndFace] = mesh.nInternalCells + np.arange(cellStartFace, cellEndFace)
    BCsM = Matrix(sp.csr_matrix((data, (row, col)), shape=(m, n)), np.zeros((m,) + phi.dimensions))
    M.b = np.vstack((M.b, BCsM.b))
    M.A = sp.vstack((M.A, BCsM.A))
    return M

if __name__ == "__main__":
    from mesh import Mesh
    mesh = Mesh.create('cases/cylinder/')
    Field.setMesh(mesh)
    T = IOField.read('U', mesh, 2.0)
    T.partialComplete()
    T.old = T.field
    res = (ddt(T, 1.) + laplacian(T, 1)).solve()
    TL = IOField('TL', res.reshape(-1,1), (1,))
    TL.partialComplete()
    TL.write(2.0)


