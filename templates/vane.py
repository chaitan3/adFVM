import numpy as np

from adFVM import config
from adFVM.compat import intersectPlane
from adFVM.density import RCF 
from adFVM import tensor
from adFVM.mesh import Mesh

from vane_obj import objective, getPlane, getWeights

# heat transfer
primal = RCF('/home/talnikar/adFVM/cases/vane/laminar/', objective=objective, fixedTimeStep=True)
#primal = RCF('/home/talnikar/adFVM/cases/vane/3d_10/', objective=objective, fixedTimeStep=True)
#primal = RCF('/home/talnikar/adFVM/cases/vane/les/', objective=objective, fixedTimeStep=True)
getPlane(primal)
getWeights(primal)

def makePerturb(param, eps=1e-6):
    def perturbMesh(fields, mesh, t):
        if not hasattr(perturbMesh, 'perturbation'):
            ## do the perturbation based on param and eps
            #perturbMesh.perturbation = mesh.getPerturbation()
            points = np.zeros_like(mesh.points)
            #points[param] = eps
            points[:] = eps*mesh.points
            #points[:] = eps
            perturbMesh.perturbation = mesh.getPointsPerturbation(points)
        return perturbMesh.perturbation
    return perturbMesh
perturb = [makePerturb(1)]
#perturb = []

parameters = 'mesh'

#def makePerturb(mid):
#    def perturb(fields, mesh, t):
#        G = 1e1*np.exp(-1e2*np.linalg.norm(mid-mesh.cellCentres[:mesh.nInternalCells], axis=1, keepdims=1)**2)
#        #rho
#        rho = G
#        rhoU = np.zeros((mesh.nInternalCells, 3))
#        rhoU[:, 0] = G.flatten()*100
#        rhoE = G*2e5
#        return rho, rhoU, rhoE
#    return perturb
#perturb = [makePerturb(np.array([-0.02, 0.01, 0.005])),
#           makePerturb(np.array([-0.08, -0.01, 0.005]))]
#
#parameters = 'source'

nSteps = 10
writeInterval = 5
sampleInterval = 5
reportInterval = 5
#nSteps = 100000
#writeInterval = 5000
#startTime = 3.0
startTime = 3.0
dt = 1e-8

#adjParams = [1e-3, 'abarbanel', None]
adjParams = [1e-3, 'entropy_jameson', None]
#adjParams = [1e-3, 'uniform', None]
