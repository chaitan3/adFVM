from pyRCF import RCF
from compute import getHTC, getIsentropicMa
import config

import matplotlib.pyplot as plt
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
for time in user.time:
    rho, rhoU, rhoE = solver.initFields(time)
    U, T, p = solver.U, solver.T, solver.p

    patches = ['pressure', 'suction']
    htc = getHTC(T, 420., patches)
    for patchID in patches:
        startFace = mesh.boundary[patchID]['startFace']
        endFace = startFace + mesh.boundary[patchID]['nFaces']
        x = mesh.faceCentres[startFace:endFace, 0]
        nFaces = x.shape[0]
        nFacesPerLayer = nFaces/nLayers
        x = x[:nFacesPerLayer]
        y = htc[patchID].reshape((nLayers, nFacesPerLayer))
        y = y.sum(axis=0)/nLayers
        plt.plot(x, y)
        plt.savefig(patchID + '.png')
        plt.clf()

    #Ma = getIsentropicMa(p, 171325.)

    
    
