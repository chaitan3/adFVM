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
locals()['reportInterval'] = 1
parameters = []

config.importModule(locals(), user.caseFile)
#print(locals().keys())
#assert all(key in locals() for key in ['primal', 'objective', 'perturb', 'nSteps', 'writeInterval', 'reportInterval', 'startTime', 'dt', 'paramters'])
assert writeInterval >= reportInterval

if not isinstance(parameters, list):
    parameters = [parameters]
if not isinstance(perturb, list):
    perturb = [perturb]
nPerturb = len(perturb)

pprint('Compiling objective')
fields = primal.getSymbolicFields()
objectiveValue = objective(fields, primal.mesh)
fields = [phi.field for phi in fields]
objectiveFunction = primal.function(fields, objectiveValue, 'objective', BCs=False, source=False)
# objective is anyways going to be a sum over all processors
# so no additional code req to handle parallel case
gradient = ad.grad(objectiveValue, fields, disconnected_inputs='warn')
objectiveGradient = primal.function(fields, gradient, 'objective_grad', BCs=False, source=False)
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
    parser.add_argument('option', nargs='?', default='orig')
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
        if parallel.rank == 0:
            timeSteps = np.loadtxt(timeStepFile)
            timeSteps = np.concatenate((timeSteps, np.array([[0, 0]])))
        else:
            timeSteps = np.zeros((nSteps+1, 2))
        parallel.mpi.Bcast(timeSteps, root=0)
        dts = timeSteps[startIndex:,1]
        writeInterval = config.LARGE

        nSims = nPerturb
    else:
        print('WTF')
        exit()

    primal.readFields(startTime)
    primal.compile()

    # restarting perturb not fully supported
    for sim in range(0, nSims):
        if user.option == 'perturb':
            perturbation = (parameters, perturb[sim])
            primal.timeStepFile = primal.mesh.case + 'timeSeries_{}.txt'.format(sim)
        else:
            perturbation = None
        result = primal.run(result=initResult, startTime=startTime, dt=dts, nSteps=nSteps, 
                            writeInterval=writeInterval, reportInterval=reportInterval, 
                            mode=user.option, startIndex=startIndex, source=source, perturbation=perturbation)
        writeResult(user.option, result/(nSteps + 1), '{}'.format(sim))
        primal.removeStatusFile()
        
