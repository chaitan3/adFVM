#!/usr/bin/python2
from __future__ import print_function

import config, parallel
from config import ad
from parallel import pprint
from field import CellField, Field, IOField

import numpy as np
import sys
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('caseFile')
user, args = parser.parse_known_args(config.args)
caseDir, caseFile = os.path.split(user.caseFile)
sys.path.append(os.path.abspath(caseDir))
caseFile = __import__(caseFile.split('.')[0])
for attr in dir(caseFile):
    if not attr.startswith('_'):
        # defines primal, objective and perturb, nSteps, writeInterval, startTime, dt
        locals()[attr] = getattr(caseFile, attr)

pprint('Compiling objective')
stackedFields = ad.matrix()
#stackedFields.tag.test_value = np.random.rand(primal.mesh.origMesh.nCells, 5).astype(config.precision)
fields = primal.unstackFields(stackedFields, CellField)
objectiveValue = objective(fields, primal.mesh)
objectiveFunction = primal.function([stackedFields], objectiveValue, 'objective', BCs=False, postpro=True)
# objective is anyways going to be a sum over all processors
# so no additional code req to handle parallel case
objectiveGradient = primal.function([stackedFields], ad.grad(objectiveValue, stackedFields), 'objective_grad', BCs=False, postpro=True)
primal.objective = objectiveFunction
primal.timeStepFile = primal.mesh.case + '{0}.{1}.txt'.format(nSteps, writeInterval)
pprint('')

def writeResult(option, result):
    globalResult = parallel.sum(result)
    resultFile = primal.resultFile
    if parallel.rank == 0:
        if option == 'perturb':
            previousResult = float(open(resultFile).readline().split(' ')[1])
            globalResult -= previousResult
        with open(resultFile, 'a') as handle:
            handle.write('{0} {1}\n'.format(option, globalResult))

if __name__ == "__main__":
    mesh = primal.mesh.origMesh
    timeStepFile = primal.timeStepFile
    statusFile = primal.statusFile

    parser = argparse.ArgumentParser()
    parser.add_argument('option')
    user = parser.parse_args(args)

    try:
        with open(statusFile, 'r') as status:
            startIndex, startTime, dt, initResult = status.readlines()
        pprint('Read status file, index =', startIndex)
        startTime = float(startTime)
        dt = float(dt)
        startIndex = int(startIndex)
        initResult = float(initResult)
    except:
        startIndex = 0
        initResult = 0.
    
    initTimeSteps = None
    if user.option == 'orig':
        dts = dt
        if parallel.rank == 0:
            try:
                initTimeSteps = np.loadtxt(timeStepFile)
            except:
                initTimeSteps = np.empty((0,2))

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
        primal.sourceTerm = perturb

    elif user.option == 'test':
        primal.initFields(startTime)
        a = np.zeros((mesh.nCells, 5))
        perturb(a, startTime)
        fields = primal.unstackFields(a, IOField)
        primal.writeFields(fields, 100.0)

        #p = np.zeros((mesh.nCells, 5))
        #fields = primal.initFields(startTime)
        #stackedFields = primal.stackFields(fields, np) 
        #result = objectiveFunction(stackedFields)
        #perturb(p, startTime)
        #print(np.sum(objectiveGradient(stackedFields)*p))
        #stackedFields += p
        #resultp = objectiveFunction(stackedFields)
        #print(resultp-result)

        #primal.adjoint = True
        #p = np.zeros((mesh.nCells, 5))
        #perturb(p, startTime)
        #writeInterval = config.LARGE
        #timeSteps, result = primal.run(startTime=startTime, dt=dt, nSteps=nSteps, writeInterval=writeInterval, objective=objectiveFunction, perturb=None)
        #timeSteps, resultp = primal.run(startTime=startTime, dt=dt, nSteps=nSteps, writeInterval=writeInterval, objective=objectiveFunction, perturb=perturb)
        #solutions = primal.run(startTime=startTime, dt=dt, nSteps=nSteps, writeInterval=writeInterval, mode='forward')
        #grad1 = np.ascontiguousarray(objectiveGradient(solutions[-1]))
        #grad2 = np.ascontiguousarray(primal.gradient(solutions[0], grad1))
        #grad3 = np.ascontiguousarray(objectiveGradient(solutions[0]))
        #grad = grad2 + grad3
        #print(np.sum(grad*p))
        #print(resultp-result)

        exit()
    else:
        print('WTF')
        exit()

    result = primal.run(result=initResult, startTime=startTime, dt=dts, nSteps=nSteps, writeInterval=writeInterval, mode=user.option, startIndex=startIndex, initTimeSteps=initTimeSteps)
    writeResult(user.option, result/(nSteps + 1))
    os.remove(statusFile)
