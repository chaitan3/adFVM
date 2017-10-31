import numpy as np
import sys
#from match import *

from adFVM import config, parallel
from adFVM.field import IOField
from adFVM.parallel import pprint
from adFVM.postpro import getHTC, getIsentropicMa, getPressureLoss, getYPlus
from adFVM.density import RCF

#config.hdf5 = True

def postprocess(solver, time, suffix=''):
    mesh = solver.mesh

    point = np.array([0.052641,-0.1,0.005])
    normal = np.array([1.,0.,0.])
    patches = ['pressure', 'suction']

    pprint('postprocessing', time)
    rho, rhoU, rhoE = solver.initFields(solver.readFields(time, suffix=suffix))
    U, T, p = solver.primitive(rho, rhoU, rhoE)
    T0 = 420.
    p0 = solver.p.phi.BC['inlet'].patch['_pt'][0,0]
    #p0 = 177000.
    print p0
    
    htc = getHTC(T, T0, patches)
    Ma = getIsentropicMa(p, p0, patches)
    print Ma
    wakeCells, pl = getPressureLoss(p, T, U, p0, point, normal)
    uplus, yplus, ustar, yplus1 = getYPlus(U, T, rho, patches)

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

    htc = IOField.boundaryField('htc' + suffix, htc, (1,))
    Ma = IOField.boundaryField('Ma' + suffix, Ma, (1,))
    uplus = IOField.boundaryField('uplus' + suffix, uplus, (3,))
    yplus = IOField.boundaryField('yplus' + suffix, yplus, (1,))
    with IOField.handle(time):
        htc.write()
        Ma.write()
        uplus.write()
        yplus.write()
        #join = '/'
        #if config.hdf5:
        #    join = '_'
        #with open(solver.mesh.getTimeDir(time) + join +  'wake' + suffix, 'w') as f:
        #    np.savez(f, wakeCells, pl)


 
if __name__ == '__main__':
    case = sys.argv[1]
    times = [float(x) for x in sys.argv[2:]]

    solver = RCF(case)
    if len(times) == 0:
        times = solver.mesh.getTimes()
    solver.readFields(times[0])
    solver.compile()

    # average
    #time = times[0]
    #postprocess(solver, time, suffix='_avg')
    #pprint()

    # instantaneous
    for index, time in enumerate(times):
        postprocess(solver, time)
        pprint()
