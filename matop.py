from field import Field
import numpy as np
import scipy.sparse as sp
import time
from parallel import pprint

class Matrix(object):
    def __init__(self, A, b=None):
        self.A = A
        m, n = A.shape
        self.b = b
        if b is None:
            self.b = np.zeros((m, 1))
        assert b.shape[0] == m

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
        if sp.issparse(b) or isinstance(b, np.ndarray):
            return self.__class__(self.A * b, self.b * b)
        else:
            raise Exception("WTF")

    def __rmul__(self, b):
        if sp.issparse(b) or isinstance(b, np.ndarray):
            return self.__class__(b * self.A, b * self.b)
        else:
            raise Exception("WTF")

    def solve(self):
        m, n = self.A.shape
        assert m == n
        return sp.linalg.spsolve(self.A, -self.b)

def laplacian(phi, DT):
    dim = phi.dimensions
    mesh = phi.mesh.origMesh
    n = mesh.nLocalCells
    m = mesh.nFaces - (mesh.nCells - mesh.nLocalCells)
    data = (mesh.areas*DT.field/mesh.deltas).flatten()
    row = np.arange(0, mesh.nFaces, dtype=np.int32)
    data = np.concatenate((data, -data[:m]))
    row = np.concatenate((row, row[:m]))
    col = np.concatenate((mesh.owner, mesh.neighbour[:m]))
    snGradOp = sp.csr_matrix((data, (row, col)), shape=(mesh.nFaces, n))
    b = np.zeros((mesh.nFaces,) + dim)
    if mesh.nFaces > m:
        b -= data[m:mesh.nFaces].reshape(-1,1)*phi.field[mesh.neighbour[m:]]
    snGradM = Matrix(snGradOp, b)
    return (sp.diags(1./mesh.volumes.flatten(), 0)*mesh.sumOp)*snGradM

def ddt(phi, dt):
    mesh = phi.mesh.origMesh
    oldPhi = phi.old[:mesh.nInternalCells]
    A = sp.eye(mesh.nInternalCells, mesh.nLocalCells)*(1./dt)
    b = -phi.old[:mesh.nInternalCells]/dt
    return Matrix(A, b)

# BC correction
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

# OLD CODE
def div(phi, U):
    if not hasattr(div, 'A'):
        internalField = phi.getInternalField()
        phi.setInternalField(internalField)
        res = op.div(phi, U)
        div.A = res.field.diff(internalField)
    return Matrix(div.A)


def hybrid(equation, boundary, fields, solver):
    start = time.time()

    names = [phi.name for phi in fields]
    shapes = [phi.getInternalField().shape for phi in fields]
    pprint('Time marching for', ' '.join(names))
    for index in range(0, len(fields)):
        fields[index].old = fields[index]
        fields[index].info()

    LHS = equation(*fields)
    internalFields = [LHS[index].solve().reshape(shapes[index]) for index in range(0, len(fields))]
    newFields = boundary(*internalFields)
    for index in range(0, len(fields)):
        newFields[index].name = fields[index].name

    end = time.time()
    pprint('Time for iteration:', end-start)
    return newFields


