from pyRCF import RCF
from compute import getHTC, getIsentropicMa, getPressureLoss
import config

from match import match_htc, match_velocity
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('case')
parser.add_argument('time', nargs='+', type=float)
user = parser.parse_args(config.args)

solver = RCF(user.case)
mesh = solver.mesh.origMesh
solver.initialize(user.time[0])

nLayers = 1
#nLayers = 200

for index, time in enumerate(user.time):
    rho, rhoU, rhoE = solver.initFields(time)
    U, T, p = solver.U, solver.T, solver.p

    patches = ['pressure', 'suction']
    htc = getHTC(T, 420., patches)
    # p = 186147, U = 67.642, T = 420, c = 410, p0 = 189718
    Ma = getIsentropicMa(p, 189718.67, patches)

    point = np.array([0.052641,-0.1,0.005])
    normal = np.array([1.,0.,0.])
    pl = getPressureLoss(p, T, U, 190000, point, normal)

    htc_args = []
    Ma_args = []
    for patchID in patches:
        startFace = mesh.boundary[patchID]['startFace']
        endFace = startFace + mesh.boundary[patchID]['nFaces']
        x = mesh.faceCentres[startFace:endFace, 0]
        nFaces = x.shape[0]
        nFacesPerLayer = nFaces/nLayers
        x = x[:nFacesPerLayer]

        y = htc[patchID].reshape((nLayers, nFacesPerLayer))
        y = y.sum(axis=0)/nLayers
        htc_args.extend([y, x])
        y = Ma[patchID].reshape((nLayers, nFacesPerLayer))
        y = y.sum(axis=0)/nLayers
        Ma_args.extend([y, x])

    htc_args.append('{}/htc_{:04d}.png'.format(user.case, index))
    Ma_args.append('{}/Ma_{:04d}.png'.format(user.case, index))
    match_htc(*htc_args) 
    match_velocity(*Ma_args)
    
