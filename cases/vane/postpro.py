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

#nLayers = 1
nLayers = 200
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

    for patchID in patches:
        nFacesPerLayer = mesh.boundary[patchID]['nFaces']/nLayers

        y = htc[patchID].reshape((nLayers, nFacesPerLayer))
        y = y.sum(axis=0)/nLayers
        HTC[patchID] += y

        y = Ma[patchID].reshape((nLayers, nFacesPerLayer))
        y = y.sum(axis=0)/nLayers
        MA[patchID] += y

    nCellsPerLayer = pl.shape[0]/nLayers
    pl = pl.reshape((nLayers, nCellsPerLayer))
    pl = pl.sum(axis=0)/nLayers
    PL += pl
   
    IOField.openHandle(time)
    uplus = IOField.boundaryField('uplus', uplus, (3,))
    uplus.write()
    yplus = IOField.boundaryField('yplus', yplus, (1,))
    yplus.write()
    IOField.closeHandle()
    pprint()

htc_args = []
Ma_args = []

for patchID in patches:
    startFace = mesh.boundary[patchID]['startFace']
    nFaces = mesh.boundary[patchID]['nFaces']
    endFace = startFace + nFaces
    nFacesPerLayer = nFaces/nLayers

    x = mesh.faceCentres[startFace:endFace, 0]
    x = x[:nFacesPerLayer]

    y = HTC[patchID]/nTimes
    htc_args.extend([y, x])
    y = MA[patchID]/nTimes
    Ma_args.extend([y, x])

y = PL/(p0*nTimes)
x = 1000*mesh.cellCentres[wakeCells[:nCellsPerLayer], 1]
wake_args = [p0, y, x]

htc_args.append('{}/htc.png'.format(case))
Ma_args.append('{}/Ma.png'.format(case))
wake_args.append('{}/wake.png'.format(case))

match_wakes(*wake_args)
match_htc(*htc_args) 
match_velocity(*Ma_args)
 
