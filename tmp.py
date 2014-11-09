from mesh import Mesh
from field import CellField
from config import ad, T, adsparse
import numpy as np
import scipy.sparse as sp

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
#
#print(a.get_value())
#
#x = ad.dscalar()
#b = a*x
#f = T.function([x], [b, b*2])
#print(f(3))
#print(a.get_value())

x = ad.dscalar()
y = ad.dscalar()
#z = ad.concatenate((x, y), axis=1)
#xt = x.reshape((-1,1))
z = T.ifelse.ifelse(ad.lt(x, y), y, x)
f = T.function([x, y], z)

X = 2.0
Y = 1.9
print(f(X, Y))

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


