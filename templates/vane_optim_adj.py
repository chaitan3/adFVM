from adFVM.density import RCF 
import sys
sys.path.append(os.path.expanduser('~/adFVM/templates'))
from vane_obj import objective, getPlane, getWeights

#config.hdf5 = True
caseDir = './'
nParam = 4
#primal = RCF(caseDir, objective='pressureLoss', objectivePLInfo={})
#primal = RCF(caseDir, objective='heatTransfer', objectiveDragInfo="pressure|suction")
primal = RCF(caseDir, timeSeriesAppend='2', fixedTimeStep=True, objective='optim', objectivePLInfo={})

#primal = RCF('/home/talnikar/adFVM/cases/vane/3d_10/', objective=objective)
#primal = RCF('/home/talnikar/adFVM/cases/vane/les/', objective=objective)
getPlane(primal)
getWeights(primal)


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
sampleInterval = 20
nSteps = 20000
writeInterval = 500
avgStart = 4000
#nSteps = 18000
#writeInterval = 1000
#avgStart = 3000
startTime = 3.001
dt = 2e-8
adjParams = [1e-3, 'entropy_jameson', None]
#runCheckpoints = 10

# definition of 1 flow through time
# 4e-4s = (0.08m)/(200m/s)

