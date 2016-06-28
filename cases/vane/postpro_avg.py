from pyRCF import RCF
from field import IOField
import config
from parallel import pprint
import sys

import numpy as np
from match import *
case = sys.argv[1]
times = [float(x) for x in sys.argv[2:]]

solver = RCF(case)
if len(times) == 0:
    times = solver.mesh.getTimes()
mesh = solver.mesh.origMesh
solver.initialize(times[0])

patches = ['pressure', 'suction']

nTimes = len(times)
for index, time in enumerate(times):
    pprint('postprocessing', time)
    rho, rhoU, rhoE = solver.initFields(time, suffix='_avg')
    U, T, p = solver.U, solver.T, solver.p
    _, _, ustar, yplus1 = getYPlus(U, T, rho, patches)
    
    for patchID in patches:
        startFace = mesh.boundary[patchID]['startFace']
        endFace = startFace + mesh.boundary[patchID]['nFaces']
        index = 0
        mult = np.logspace(0, 2, 100)
        points = mesh.faceCentres[startFace + index] \
               - mesh.normals[startFace + index]*yplus1[patchID][index]*mult.reshape(-1,1)
        field = U.interpolate(points)/ustar[patchID][index]
        field = ((field**2).sum(axis=1))**0.5
        plt.plot(mult, field)
        plt.show()

    pprint()
