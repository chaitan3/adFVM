#!/usr/bin/python2

from field import CellField
from mesh import Mesh

mesh = Mesh('tests/forwardStep/')
U = CellField.read('U', mesh, 0.000771531)
p = CellField.read('p', mesh, 0.000771531)
T = CellField.read('T', mesh, 0.000771531)
