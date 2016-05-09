from field import Field, IOField
from mesh import Mesh
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
    def create(self, m, n):
        A = PETSc.Mat()
        A.create(PETSc.COMM_WORLD)
        A.setSizes(((m, PETSc.DECIDE), (n, PETSc.DECIDE)))
        A.setType('aij')
        A.setPreallocationNNZ(6) 

        x, b = A.createVecs()
        return self(A, b)

    def __add__(self, b):
        if isinstance(b, Matrix):
            return self.__class__(self.A + b.A, self.b + b.b)
        elif isinstance(b, Field):
            return self.__class__(self.A, self.b + b.field)
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
        m, n = self.A.shape
        assert m == n
        return sp.linalg.spsolve(self.A, -self.b)

def laplacian(phi, DT):
    dim = phi.dimensions
    mesh = phi.mesh.origMesh
    #n = mesh.nLocalCells
    #m = mesh.nFaces - (mesh.nCells - mesh.nLocalCells)
    l = mesh.nFaces
    m = mesh.nInternalFaces
    n = mesh.nInternalCells
    o = mesh.nFaces - (mesh.nCells - mesh.nLocalCells)

    snGradM = Matrix.create(l, n)
    snGradOp, snGradb = snGradM.A, snGradM.b

    il, ih = snGradOp.getOwnershipRange()
    jl, jh = snGradOp.getOwnershipRangeColumn()
    #data = (mesh.areas*DT.field/mesh.deltas).flatten()
    data = (mesh.areas/mesh.deltas).flatten()
    row = np.arange(0, l, dtype=np.int32)
    data = np.concatenate((-data, data[:m], data[l:]))
    row = np.concatenate((row, row[:m], row[l:]))
    procCols = mesh.neighbour[l:]
    procRanges = snGradOp.getOwnershipRanges()
    for patchID in phi.mesh.remotePatches:
        patch = mesh.boundary[patchID]
        startFace = patch['startFace']
        endFace = startFace + patch['nFaces']
        proc = patch['neighbProcNo']
        procCols[startFace:endFace] += -jl + procRanges[proc]
    col = np.concatenate((mesh.owner, mesh.neighbour[:m], procCols))
    snGradOp.setValuesRCV(il + row.reshape(-1,1), jl + col.reshape(-1,1), data.reshape(-1,1))
    snGradOp.assemble()

    indices = np.arange(m, o).astype(np.int32)
    data = data[m:o].reshape(-1,1)*phi.field[mesh.neighbour[m:o]]
    snGradb.setValues(il + indices, data)
    snGradb.assemble()
    
    sumOp = Matrix.create(n, l).A
    il, ih = sumOp.getOwnershipRange()
    jl, jh = sumOp.getOwnershipRangeColumn()
    indices = mesh.sumOp.indices
    indptr = mesh.sumOp.indptr
    data = mesh.sumOp.data
    sumOp.setValuesIJV(indptr, il + indices, data)
    sumOp.assemble()
    M = sumOp * snGradM

    return (sp.diags(1./mesh.volumes.flatten(), 0)*mesh.sumOp)*snGradM

def ddt(phi, dt):
    mesh = phi.mesh.origMesh
    n = mesh.nInternalCells
    M = Matrix.create(n, n)
    A, b = M.A, M.b

    il, ih = A.getOwnershipRange()
    diag = np.arange(0,n).astype(np.int32).reshape(-1,1)
    v = np.ones_like(diag)/dt
    A.setValuesRCV(il + diag, il + diag, v)
    A.assemble()

    oldPhi = phi.old[:n]
    b.setValues(il + diag[:,0], -oldPhi/dt)
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
    mesh = Mesh.create('cases/cylinder/')
    Field.setMesh(mesh)
    T = IOField.read('T', mesh, 2.0)
    T.partialComplete()
    T.old = T.field
    ddt(T, 1.)
    laplacian(T, 1)


