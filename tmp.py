from mesh import Mesh
from field import CellField
from config import ad, T, adsparse
import numpy as np
import scipy.sparse as sp
from mpi4py import MPI

#mesh = Mesh('tests/cylinder')
#
#class empty: pass
#solver = empty()
#solver.mesh = mesh
#CellField.setSolver(solver)
#
#U = CellField.read('U', 2.0)

#A = np.random.rand(100, 3)
#n = np.random.rand(100, 1)
#B = ad.dmatrix()
#C = ad.sum(B * n, axis=1).reshape((-1, 1))
#f = T.function([B], C)
#print(f(A).shape)

#mesh = Mesh('tests/forwardStep')
#A = np.random.rand(mesh.nFaces, 3)
#B = ad.dmatrix()
#C = adsparse.basic.dot(mesh.sumOp, B)
#f = T.function([B], C)
#g = T.function([B], T.gradient.grad(ad.sum(C), B))
#print(f(A))
#print(g(A))

#a = T.shared(np.float64(1.))
#print(a.get_value())
#
#x = ad.dvector()
#b = a*x
#f = T.function([x], ad.abs_(b))
#print(f(np.array([1,-1])))

#x = ad.dscalar()
#y = ad.dscalar()
##z = ad.concatenate((x, y), axis=1)
##xt = x.reshape((-1,1))
#z = T.ifelse.ifelse(ad.lt(x, y), y, x)
#f = T.function([x, y], z)
#
#X = 2.0
#Y = 1.9
#print(f(X, Y))

#class Field(object):
#    def __init__(self, field):
#        self.field = ad.alloc(np.float64(0.), *(100, 3))
#        ad.set_subtensor(self.field[:50], field)
#        ad.set_subtensor(self.field[50:], self.field[:50])
#
#X = ad.dmatrix()
##phi = Field(X)
##Y = phi.field
#A = ad.alloc(np.float64(0.), *(100, 3))
#A = ad.set_subtensor(A[:50], X)
#A = ad.set_subtensor(A[50:], A[:50])
#
#f = T.function([X], A)

#mesh = Mesh('tests/forwardStep')
#x = ad.dmatrix()
#y = ad.dmatrix()
#z = x[mesh.owner]*mesh.weights + y[mesh.neighbour]*(1-mesh.weights)
#f = T.function([x, y], z)
#
#X = np.random.rand(mesh.nCells, 1)
#Y = np.random.rand(mesh.nCells, 1)
#print(f(X, Y))

#mpi = MPI.COMM_WORLD
#rank = mpi.Get_rank()
#print(rank)
#x = ad.dscalar()
#y = ad.dscalar()
#
#X = np.random.rand(1)[0]
#other = 1 - rank
#if rank == 0:
#    z = x**2 + y
#    mpi.send(X, dest=other)
#    Y = mpi.recv(source=other)
#    grad = 2*X + 3*X**2*Y
#else:
#    z = x * y**3
#    Y = mpi.recv(source=other)
#    mpi.send(X, dest=other)
#    grad = 1 + Y**3
#f = T.function([x, y], z)
#g = T.function([x, y], T.grad(z, x))
#h = T.function([x, y], T.grad(z, y))
#print(rank, X, Y)
#print(rank, f(X, Y))
#diff = h(X, Y)
#if rank == 0:
#    mpi.send(diff, dest=other)
#    data = mpi.recv(source=other)
#else:
#    data = mpi.recv(source=other)
#    mpi.send(diff, dest=other)
#
#print('grad ', rank, g(X, Y) + data, grad)

#x = ad.dmatrix()
#n = 0
#y = ad.alloc(np.float64(0.), *(200 + n, 3))
#y = ad.set_subtensor(y[:100], x[:100])
#y = ad.set_subtensor(y[200:], x[100:])
#f = T.function([x], y)
#a = np.random.rand(100 + n, 3)
#print a
#print f(a)
#


#mpi = MPI.COMM_WORLD
#rank = mpi.Get_rank()
#
#other = 1 - rank
#if rank == 0:
#    x = 2*np.ones(100, np.int32)
#    e = mpi.Isend(x, dest=other)
#else:
#    x = np.empty(200, np.int32)
#    e = mpi.Irecv(x, source=other)
#status = MPI.Status()
#MPI.Request.Wait(e, status)
#print x
#print status.Get_count()

#mpi = MPI.COMM_WORLD
#rank = mpi.Get_rank()
#bufsize = (2, 6)
#order = 'F'
#
#other = 1 - rank
#if rank == 0:
#    x = np.array([[3, 1, 2], [1, 5, 4]], order=order)
#    req = mpi.Isend(x, dest=other)
#else:
#    x = np.zeros(bufsize, np.int64, order=order)
#    req = mpi.Irecv(x, source=other)
#MPI.Request.Wait(req)
#print rank, x

#a = np.random.rand(10, 3)
#b = np.random.rand(10, 3)
#c = np.cross(a,b)
#print c.flags

#x = ad.matrix()
#z = ad.matrix()
#y = ad.sum(x*z)
#g = T.function([x, z], ad.grad(y, x))
#a = np.random.rand(100, 10)
#b = np.random.rand(100, 10)
#print a.flags
#print b.flags
#print g(a, b).flags

#x = ad.matrix()
#y = ad.TensorType(T.config.floatX, broadcastable=[False, True])()
#f = T.function([x,y], x*y)
#y = np.random.rand(10, 1)
#g = T.function([x], x*y)
#a = np.random.rand(10, 3)
#b = np.random.rand(10, 1)
#print a
#print b
#print a*b
#print g(a)
#print f(a, b)

#c = 1e-300
#x = ad.matrix()
#f = T.function([x], ad.switch(ad.lt(x, 1.), x - c, x + c))
#b = np.zeros((10, 3))
#print b
#print f(b)


#x = ad.TensorType(T.config.floatX, broadcastable=[False, True])()
#print x.broadcastable
#y = x[2:4]
#z = x[[2,3]]
#print y.broadcastable
#print y.broadcastable

import parallel
from op import grad
from interp import central
from pyRCF import RCF
from field import IOField
case = 'tests/cylinder'
solver = RCF(case, CFL=0.2, timeIntegrator='euler')
mesh = solver.mesh
meshC = mesh
meshP = mesh.paddedMesh
mesh = mesh.origMesh
meshO = meshP.origMesh


paddedStackedFields = ad.matrix()
rho, rhoU, rhoE = solver.unstackFields(paddedStackedFields, CellField)
rhoPF = central(rho, meshP)
gradRho = grad(rhoPF, ghost=True)
function = T.function([paddedStackedFields], [gradRho.field, rhoPF.field])

stackedFields = ad.matrix()
rho, rhoU, rhoE = solver.unstackFields(stackedFields, CellField)
rhoF = central(rho, meshC)
function2 = T.function([stackedFields], rhoF.field)

fields = solver.initFields(2.0)
stackedFields = solver.stackFields(fields, np)
parallel.getOrigRemoteCells(stackedFields, meshC)
paddedStackedFields = parallel.getRemoteCells(stackedFields, meshC)
grad, faceP = function(paddedStackedFields)
face = function2(stackedFields)

nLocalBoundaryFaces = mesh.nLocalCells - mesh.nInternalCells
nLocalRemoteBoundaryFaces = mesh.nCells - mesh.nLocalCells
nLocalInternalFaces = mesh.nInternalFaces + nLocalRemoteBoundaryFaces
remoteGhostStartFace = meshP.origMesh.nInternalFaces + nLocalBoundaryFaces

#face = mesh.areas
#faceP = meshP.origMesh.areas
faceRemote = faceP.copy()
internalCursor = nLocalInternalFaces
boundaryCursor = remoteGhostStartFace
exchanger = parallel.Exchanger()
for patchID in meshC.remotePatches:
    nInternalFaces = len(meshP.remoteFaces['internal'][patchID])
    local, remote, tag = meshC.getProcessorPatchInfo(patchID)

    exchanger.exchange(remote, face[meshP.localRemoteFaces['internal'][patchID]], faceRemote[internalCursor:internalCursor + nInternalFaces], tag)
    internalCursor += nInternalFaces
    nBoundaryFaces = len(meshP.remoteFaces['boundary'][patchID])
    tag += len(meshC.origPatches) + 1
    exchanger.exchange(remote, face[meshP.localRemoteFaces['boundary'][patchID]], faceRemote[boundaryCursor:boundaryCursor + nBoundaryFaces], tag)
    boundaryCursor += nBoundaryFaces
exchanger.wait()

start, end = meshP.origMesh.nInternalFaces, remoteGhostStartFace
diff = np.abs(faceP[start:end]-faceRemote[start:end])
print 'local boundary', np.max(diff)

internalCursor = nLocalInternalFaces
boundaryCursor = remoteGhostStartFace
for patchID in meshC.remotePatches:
    nInternalFaces = len(meshP.remoteFaces['internal'][patchID])
    start, end = internalCursor, internalCursor + nInternalFaces
    diff = np.abs(faceP[start:end]-faceRemote[start:end])
    print 'internal', patchID, np.max(diff), np.argmax(diff)
    internalCursor += nInternalFaces
    nBoundaryFaces = len(meshP.remoteFaces['boundary'][patchID])
    start, end = boundaryCursor, boundaryCursor + nBoundaryFaces
    diff = np.abs(faceP[start:end]-faceRemote[start:end])
    print 'boundary', patchID, np.max(diff), np.argmax(diff)
    boundaryCursor += nBoundaryFaces

gradRemote = grad.copy()
parallel.getOrigRemoteCells(gradRemote, meshC)
for patchID in meshC.remotePatches:
    patch = mesh.boundary[patchID]
    startFace = patch['startFace']
    endFace = startFace + patch['nFaces']

    start = mesh.nInternalCells + startFace - mesh.nInternalFaces
    end = mesh.nInternalCells + endFace - mesh.nInternalFaces
    g1 = grad[start:end]
    g2 = gradRemote[start:end]
    diff = np.abs(np.linalg.norm(g1, axis=1)-np.linalg.norm(g2, axis=1))
    maxd = np.argmax(diff)
    print 'grad', patchID, np.max(diff)
    #P = T.function([], meshP.sumOp)()
    #C = T.function([], meshC.sumOp)()
    #x = C*face
    #y = P*faceP
    #i = np.argmax(y[:mesh.nInternalCells]-x)
    ##import pdb;pdb.set_trace()

from config import adsparse
import config
rho = ad.matrix()
rhoF = rho[meshP.owner] + rho[meshP.neighbour]
rhoC = adsparse.basic.dot(adsparse.sqr(meshP.sumOp), rhoF)
J = T.function([rho], ad.grad(rhoC[:meshC.nInternalCells].sum(), rho))

rhoD = ad.bcalloc(config.precision(0.), (mesh.nCells, 1))
rhoD = ad.set_subtensor(rhoD[:meshC.nInternalCells],rhoC[:meshC.nInternalCells])
rhoD = ad.set_subtensor(rhoD[meshC.nLocalCells:meshC.nCells], rhoC[meshC.nInternalCells:])
rhoG = rhoD[meshC.owner] + rhoD[meshC.neighbour]
rhoE = adsparse.basic.dot(adsparse.sqr(meshC.sumOp), rhoG)
K = T.function([rho], ad.grad(rhoE.sum(), rho))

rho = np.ones((meshO.nCells, 1))
J = J(rho)
J = parallel.getAdjointRemoteCells(J, meshC)
K = K(rho)
K = parallel.getAdjointRemoteCells(K, meshC)

for patchID in meshC.remotePatches:
    start = mesh.boundary[patchID]['startFace']
    end = start + mesh.boundary[patchID]['nFaces']
    print patchID, J[mesh.owner[start:end]].ravel()
    print patchID, K[mesh.owner[start:end]].ravel()

