import numpy as np
import sys, os

from adFVM import config
from adFVM.config import ad
from adFVM.compat import norm, intersectPlane
from adFVM.density import RCF 

config.hdf5 = True
caseDir = './'
#primal = RCF(caseDir, objective='pressureLoss', objectivePLInfo={})
#primal = RCF(caseDir, objective='heatTransfer', objectiveDragInfo="pressure|suction")
primal = RCF(caseDir, fixedTimeStep=True, objective='optim', objectivePLInfo={}, objectiveDragInfo="pressure|suction")
nParam = 4

# pressure loss
def getPlane(solver):
    point = np.array([0.052641,-0.1,0.005])
    normal = np.array([1.,0.,0.])
    ptin = 175158.
    interCells, interArea = intersectPlane(solver.mesh, point, normal)
    return {'cells':interCells.astype(np.int32), 
            'areas': interArea, 
            'normal': normal, 
            'ptin': ptin
           }
primal.defaultConfig["objectivePLInfo"] = getPlane(primal)
    
def makePerturb(index):
    def perturbMesh(fields, mesh, t):
        if not hasattr(perturbMesh, 'perturbation'):
            perturbMesh.perturbation = mesh.getPerturbation(caseDir + 'grad{}/'.format(index))
        return perturbMesh.perturbation
    return perturbMesh
perturb = []
for index in range(0, nParam):
    perturb.append(makePerturb(index))

parameters = 'mesh'
reportInterval = 1
#nSteps = 200000
#writeInterval = 10000
#avgStart = 10000
#sampleInterval = 100
nSteps = 10
writeInterval = 5
avgStart = 0
sampleInterval = 1
startTime = 3.0
dt = 2e-8

# definition of 1 flow through time
# 4e-4s = (0.08m)/(200m/s)

