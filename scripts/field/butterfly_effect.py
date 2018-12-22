#!/usr/bin/python
import sys, os
import numpy as np

from adFVM import config
from adFVM.field import Field, IOField
from adFVM.mesh import Mesh

case1, case2 = sys.argv[1:]
mesh = Mesh.create(case1)
times = mesh.getTimes()
Field.setMesh(mesh)

fields = ['p', 'T', 'U']
for name in fields:
    phimax = -1
    for time in times:
        mesh.case = case1
        phi1 = IOField.read(name, mesh, time)
        phi1.partialComplete()
        if phimax < 0:
            phimax = phi1.field.max()
        mesh.case = case2
        phi2 = IOField.read(name, mesh, time)
        phi2.partialComplete()
        phi1.name += '_diff'
        phi1.field = (phi2.field-phi1.field)/phimax
        phi1.write(time)

