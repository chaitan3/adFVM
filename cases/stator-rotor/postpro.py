from pyRCF import RCF
from field import IOField
from compute import getHTC, getIsentropicMa, getPressureLoss, getYPlus
import config

import numpy as np
from match import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('case')
parser.add_argument('time', nargs='+', type=float)
user = parser.parse_args(config.args)

solver = RCF(user.case)
mesh = solver.mesh.origMesh
solver.initialize(user.time[0])

#nLayers = 1
nLayers = 200

surface = 'blade'
#surface = 'blade0'
T0 = 365.
p0 = 122300.
#T0 = 409.
#p0 = 184900.
#surface = 'nozzle'
get_profile(surface)

for index, time in enumerate(user.time):
    rho, rhoU, rhoE = solver.initFields(time)
    U, T, p = solver.U, solver.T, solver.p
    
    point = np.array([0.052641,-0.1,0.005])
    normal = np.array([1.,0.,0.])
    patches = [surface + '_pressure', surface + '_suction']

    htc = getHTC(T, T0, patches)
    Ma = getIsentropicMa(p, p0, patches)
    wakeCells, pl = getPressureLoss(p, T, U, p0, point, normal)
    yplus = getYPlus(U, T, rho, patches)

    htc_args = []
    Ma_args = []
    for patchID in patches:
        startFace = mesh.boundary[patchID]['startFace']
        endFace = startFace + mesh.boundary[patchID]['nFaces']
        x = mesh.faceCentres[startFace:endFace, [0,1]]
        nFaces = x.shape[0]
        nFacesPerLayer = nFaces/nLayers
        x = x[:nFacesPerLayer]

        y = htc[patchID].reshape((nLayers, nFacesPerLayer))
        y = y.sum(axis=0)/nLayers
        htc_args.extend([y, x])
        y = Ma[patchID].reshape((nLayers, nFacesPerLayer))
        y = y.sum(axis=0)/nLayers
        Ma_args.extend([y, x])

    nCellsPerLayer = pl.shape[0]/nLayers
    pl = pl.reshape((nLayers, nCellsPerLayer))
    pl = pl.sum(axis=0)/nLayers
    y = pl/p0
    x = mesh.cellCentres[wakeCells[:nCellsPerLayer], 1]
    wake_args = [y, x]

    htc_args.append('{}/{}_htc_{:04d}.png'.format(user.case, surface, index))
    Ma_args.append('{}/{}_Ma_{:04d}.png'.format(user.case, surface, index))
    wake_args.append('{}/{}_wake_{:04d}.png'.format(user.case, surface, index))
    match_wakes(*wake_args)
    match_htc(*htc_args) 
    match_velocity(*Ma_args)
    
    IOField.openHandle(solver.mesh.case, time)
    yplus = IOField.boundaryField('yplus', yplus, (1,))
    yplus.write(time)
    IOField.closeHandle()
