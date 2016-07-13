import sys
import numpy as np
from match import *

from adFVM import config
from adFVM.field import IOField
from adFVM.parallel import pprint
from adFVM.postpro import getHTC, getIsentropicMa, getPressureLoss, getYPlus
from adFVM.density import RCF

def postpro(solver, time, suffix=''):

    #nLayers = 1
    nLayers = 200

    surface = 'blade'
    #surface = 'blade0'
    T0 = 365.
    p0 = 122300.
    point = np.array([0.081 + 0.023*0.03973,-0.06,0.005])
    normal = np.array([1.,0.,0.])
    #T0 = 409.
    #p0 = 184900.
    #surface = 'nozzle'
    # valid?
    #point = np.array([0.052641,-0.1,0.005])
    #normal = np.array([1.,0.,0.])
    get_profile(surface)

    patches = [surface + '_pressure', surface + '_suction']
    #patches = ['blade_pressure', 'blade_suction', 'blade0_pressure', 'blade0_suction', 'nozzle_pressure', 'nozzle_suction']

    pprint('postprocessing', time)
    rho, rhoU, rhoE = solver.initFields(time, suffix=suffix)
    U, T, p = solver.U, solver.T, solver.p
    
    htc = getHTC(T, T0, patches)
    Ma = getIsentropicMa(p, p0, patches)
    wakeCells, pl = getPressureLoss(p, T, U, p0, point, normal)
    uplus, yplus, _, _ = getYPlus(U, T, rho, patches)

    htc = IOField.boundaryField('htc' + suffix, htc, (1,))
    Ma = IOField.boundaryField('Ma' + suffix, Ma, (1,))
    uplus = IOField.boundaryField('uplus' + suffix, uplus, (3,))
    yplus = IOField.boundaryField('yplus' + suffix, yplus, (1,))
    with IOField.handle(time):
        htc.write()
        Ma.write()
        uplus.write()
        yplus.write()

if __name__ == '__main__':
    case = sys.argv[1]
    times = [float(x) for x in sys.argv[2:]]

    solver = RCF(case)
    if len(times) == 0:
        times = solver.mesh.getTimes()
    solver.readFields(times[0])

    # plot over surface normals
    #for patchID in patches:
    #    startFace = mesh.boundary[patchID]['startFace']
    #    endFace = startFace + mesh.boundary[patchID]['nFaces']
    #    index = 0
    #    mult = np.logspace(0, 2, 100)
    #    points = mesh.faceCentres[startFace + index] \
    #           - mesh.normals[startFace + index]*yplus1[patchID][index]*mult.reshape(-1,1)
    #    field = U.interpolate(points)/ustar[patchID][index]
    #    field = ((field**2).sum(axis=1))**0.5
    #    plt.plot(mult, field)
    #    plt.show()

    # average
    #time = times[0]
    #postprocess(solver, time, suffix='_avg')
    #pprint()

    # instantaneous
    for index, time in enumerate(times):
        postprocess(solver, time)
        pprint()

