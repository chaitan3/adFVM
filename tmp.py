from mesh import Mesh
from field import CellField
from config import ad, T
import numpy as np

#mesh = Mesh('tests/cylinder')
#
#class empty: pass
#solver = empty()
#solver.mesh = mesh
#CellField.setSolver(solver)
#
#U = CellField.read('U', 2.0)

A = np.random.rand(100, 3)
n = np.random.rand(100, 1)
B = ad.dmatrix()
C = ad.sum(B * n, axis=1).reshape((-1, 1))
f = T.function([B], C)
print(f(A).shape)
