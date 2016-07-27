#!/usr/bin/python2
from __future__ import print_function

from adFVM import config, parallel
from adFVM.config import ad
from adFVM.parallel import pprint
from adFVM.field import CellField

import numpy as np
import sys
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('caseFile')
user, args = parser.parse_known_args(config.args)

source = lambda *args: []
perturb = []

caseDir, caseFile = os.path.split(user.caseFile)
sys.path.append(os.path.abspath(caseDir))
caseFile = __import__(caseFile.split('.')[0])
for attr in dir(caseFile):
    if not attr.startswith('_'):
        # defines primal, objective and perturb, nSteps, writeInterval, startTime, dt
        locals()[attr] = getattr(caseFile, attr)

if not isinstance(perturb, list):
    perturb = [perturb]
nPerturb = len(perturb)

pprint('Compiling objective')
stackedFields = ad.matrix()
#stackedFields.tag.test_value = np.random.rand(primal.mesh.origMesh.nCells, 5).astype(config.precision)
fields = primal.unstackFields(stackedFields, CellField)
objectiveValue = objective(fields, primal.mesh)
objectiveFunction = primal.function([stackedFields], objectiveValue, 'objective', BCs=False, source=False)
# objective is anyways going to be a sum over all processors
# so no additional code req to handle parallel case
objectiveGradient = primal.function([stackedFields], ad.grad(objectiveValue, stackedFields), 'objective_grad', BCs=False, source=False)
primal.objective = objectiveFunction
primal.timeStepFile = primal.mesh.case + '{0}.{1}.txt'.format(nSteps, writeInterval)
pprint('')

def writeResult(option, result, info=''):
    globalResult = parallel.sum(result)
    resultFile = primal.resultFile
    if parallel.rank == 0:
        if option == 'perturb':
            previousResult = float(open(resultFile).readline().split(' ')[2])
            globalResult -= previousResult
        with open(resultFile, 'a') as handle:
            if len(info) > 0:
                handle.write('{} {} {}\n'.format(option, info, globalResult))
            else:
                handle.write('{} {}\n'.format(option, globalResult))

if __name__ == "__main__":
    mesh = primal.mesh.origMesh
    timeStepFile = primal.timeStepFile

    parser = argparse.ArgumentParser()
    parser.add_argument('option')
    user = parser.parse_args(args)

    if parallel.mpi.bcast(os.path.exists(primal.statusFile), root=0):
        startIndex, startTime, dt, initResult = primal.readStatusFile()
        pprint('Read status file, index =', startIndex)
    else:
        startIndex = 0
        initResult = 0.
    
    if user.option == 'orig':
        dts = dt
        nSims = 1

    elif user.option == 'perturb':
        pprint('Perturbing fields')
        if parallel.rank == 0:
            timeSteps = np.loadtxt(timeStepFile)
            timeSteps = np.concatenate((timeSteps, np.array([[0, 0]])))
        else:
            timeSteps = np.zeros((nSteps+1, 2))
        parallel.mpi.Bcast(timeSteps, root=0)
        dts = timeSteps[startIndex:,1]
        writeInterval = config.LARGE

        nSims = nPerturb

    elif user.option == 'test':
        #primal.adjoint = True
        primal.readFields(startTime)
        primal.compile()
        #primal.initFields(startTime)
        #rho, rhoU, rhoE = perturb(mesh)
        #U, T, p = primal.U, primal.T, primal.p
        #U.field *= 0
        #T.field *= 0
        #p.field *= 0
        #U.field[:mesh.nInternalCells] = rhoU
        #T.field[:mesh.nInternalCells] = rhoE
        #p.field[:mesh.nInternalCells] = rho
        #for phi in [U, T, p]:
        #    phi.write(100.0)

        result = primal.run(startTime=startTime, dt=dt, nSteps=nSteps, mode='orig')
        forward = primal.forward
        forward.values[forward.source:forward.source+3] = perturb(mesh)
        resultp = primal.run(startTime=startTime, dt=dt, nSteps=nSteps, mode='orig')
        #solutions = primal.run(startTime=startTime, dt=dt, nSteps=nSteps, writeInterval=writeInterval, mode='forward')
        #grad1 = np.ascontiguousarray(objectiveGradient(solutions[-1]))
        #grad2 = np.ascontiguousarray(primal.gradient(solutions[0], grad1))
        #grad3 = np.ascontiguousarray(objectiveGradient(solutions[0]))
        #grad = grad2 + grad3
        #print(np.sum(grad*p))
        print(result, resultp)
        print(resultp-result)

        exit()
    else:
        print('WTF')
        exit()

    primal.readFields(startTime)
    primal.compile()

    # restarting perturb not fully supported
    for sim in range(0, nSims):
        if user.option == 'perturb':
            source = perturb[sim]
        result = primal.run(result=initResult, startTime=startTime, dt=dts, nSteps=nSteps, writeInterval=writeInterval, mode=user.option, startIndex=startIndex, source=source)
        writeResult(user.option, result/(nSteps + 1), '{}'.format(sim))
        primal.removeStatusFile()
        
