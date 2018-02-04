import numpy as np

from adFVM import config
from adFVM.compat import intersectPlane
from adFVM.density import RCF 
from adpy import tensor
from adFVM.mesh import Mesh


from adFVM.objectives.vane import objective, getPlane, getWeights

primal = RCF('/home/talnikar/adFVM/cases/vane/laminar/test_matop/', objective=objective, fixedTimeStep=True)
#primal = RCF('/home/talnikar/adFVM/cases/vane/3d_10/', objective=objective, fixedTimeStep=True)
#primal = RCF('/home/talnikar/adFVM/cases/vane/les/', objective=objective, fixedTimeStep=True)
getPlane(primal)
getWeights(primal)

parameters = 'mesh'

def makePerturb(mid):
    def perturb(fields, mesh, t):
        G = 1e0*np.exp(-1e2*np.linalg.norm(mid-mesh.cellCentres[:mesh.nInternalCells], axis=1, keepdims=1)**2)
        #rho
        rho = G
        rhoU = np.zeros((mesh.nInternalCells, 3), config.precision)
        rhoU[:, 0] = G.flatten()*100
        rhoE = G*2e5
        return rho, rhoU, rhoE
    return perturb
perturb = [makePerturb(np.array([-0.02, 0.01, 0.005], config.precision)),
           makePerturb(np.array([-0.08, -0.01, 0.005], config.precision))]

parameters = 'source'

nSteps = 1
writeInterval = 1
startTime = 4.0
dt = 1e-8

#adjParams = [1e-3, 'abarbanel', None]
adjParams = [1e2, 'entropy_jameson', None]
#adjParams = [1e-2, 'entropy_jameson', None]
#adjParams = [1e-3, 'uniform', None]
