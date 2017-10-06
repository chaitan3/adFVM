import numpy as np

from adFVM import config
from adFVM.compat import intersectPlane
from adFVM.density import RCF 
from adFVM import tensor
from adFVM.mesh import Mesh

   
def objective(fields, solver):
    U, T, p = fields
    mesh = solver.mesh.symMesh
    # then elemwise
    def _combine(T):
        return (T*1).sum()
    return tensor.Tensorize(_combine)(mesh.nInternalCells)(T)[0]

#primal = RCF('./', objective=objective, fixedTimeStep=True)
primal = RCF('/home/talnikar/adFVM/cases/box/box_1/', objective=objective, fixedTimeStep=True)

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
#perturb = [makePerturb(1)]
perturb = []

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
writeInterval = 10
startTime = 0.0
dt = 1e-8

#adjParams = [1e-3, 'abarbanel', None]
