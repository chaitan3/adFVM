from pyRCF import RCF
from field import IOField
from compute import getHTC, getIsentropicMa, getPressureLoss, getYPlus
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

# p = 186147, U = 67.642, T = 420, c = 410, p0 = 189718
T0 = 420.
p0 = 212431.
#p0 = 189718.8
point = np.array([0.052641,-0.1,0.005])
normal = np.array([1.,0.,0.])
patches = ['pressure', 'suction']

HTC = {}
MA = {}
for patchID in patches:
    HTC[patchID] = 0.
    MA[patchID] = 0.
PL = 0.

nTimes = len(times)
for index, time in enumerate(times):
    pprint('postprocessing', time)
    rho, rhoU, rhoE = solver.initFields(time)
    U, T, p = solver.U, solver.T, solver.p
    
    htc = getHTC(T, T0, patches)
    Ma = getIsentropicMa(p, p0, patches)
    wakeCells, pl = getPressureLoss(p, T, U, p0, point, normal)
    uplus, yplus = getYPlus(U, T, rho, patches)

    IOField.openHandle(time)
    htc = IOField.boundaryField('htc', htc, (1,))
    htc.write()
    Ma = IOField.boundaryField('Ma', Ma, (1,))
    Ma.write()
    uplus = IOField.boundaryField('uplus', uplus, (3,))
    uplus.write()
    yplus = IOField.boundaryField('yplus', yplus, (1,))
    yplus.write()
    IOField.closeHandle()
    pprint()

    # HOW TO SAVE PL?
