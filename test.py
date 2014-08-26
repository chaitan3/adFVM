#!/usr/bin/python2
from __future__ import print_function

from mesh import Mesh
from field import CellField
from ops import interpolate
import time

case = 'tests/cylinder/'
mesh = Mesh(case)
U = CellField.read('U', mesh, 2)
p = CellField.read('p', mesh, 2)
Uf = interpolate(U).field
pf = interpolate(p).field

print(mesh.sumOp.shape)
print(pf.shape)
print(Uf.shape)

x = lambda t: mesh.sumOp * pf
y = lambda t: mesh.sumOp * Uf

