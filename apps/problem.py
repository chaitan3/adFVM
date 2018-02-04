#!/usr/bin/python2 -u
from __future__ import print_function

from adFVM import config, parallel
from adFVM.parallel import pprint

import numpy as np
import sys
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('caseFile')
user, args = parser.parse_known_args(config.args)

source = lambda *args: [0.]*len(args[0])
perturb = []
reportInterval = 1
sampleInterval = 1
viscousInterval = 1
parameters = []
adjParams = [0.0, None, None]
avgStart = 0
runCheckpoints = 2**30

config.importModule(locals(), user.caseFile)
#print(locals().keys())
#assert all(key in locals() for key in ['primal', 'objective', 'perturb', 'nSteps', 'writeInterval', 'reportInterval', 'startTime', 'dt', 'paramters'])
assert nSteps >= writeInterval
assert writeInterval >= reportInterval
assert (writeInterval % reportInterval) == 0
assert writeInterval >= sampleInterval
assert (writeInterval % sampleInterval) == 0
assert writeInterval >= viscousInterval
assert (writeInterval % viscousInterval) == 0

if not isinstance(parameters, list):
    parameters = [parameters]
if not isinstance(perturb, list):
    perturb = [perturb]
nPerturb = len(perturb)

primal.timeStepFile = primal.mesh.case + '{0}.{1}.txt'.format(nSteps, writeInterval)
pprint('')

def writeResult(option, result, info='-', timeSeriesFile=None):

    globalResult = [res/(nSteps-avgStart) for res in result]
    resultFile = primal.resultFile
    if parallel.rank == 0:
        noise = 0
        if option == 'perturb':
            previousResult = float(open(resultFile).readline().split(' ')[3])
            globalResult[0] -= previousResult
        noise = [0. for res in result]
        if timeSeriesFile:
            import ar
            timeSeries = np.loadtxt(timeSeriesFile,ndmin=1)[avgStart::sampleInterval]
            if len(timeSeries.shape) == 1:
                timeSeries = timeSeries.reshape(-1,1)
            for index in range(0, len(result)):
                noise[index] = (ar.arsel(timeSeries[:, index]).mu_sigma**2)[0]
        with open(resultFile, 'a') as handle:
            for index in range(0, len(result)):
                handle.write('{} {} {} {} {}\n'.format(option, index, info, globalResult[index], noise[index]))

if __name__ == "__main__":
    mesh = primal.mesh
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
        dts = timeSteps[:,1]

        nSims = nPerturb
    else:
        print('WTF')
        exit()

    primal.readFields(startTime)
    primal.compile()

    # restarting perturb not fully supported
    for sim in range(0, nSims):
        if user.option == 'perturb':
            if config.gpu:
                pprint('On GPU for multiple perturbations will give wrong results')
            # single set of parameters, but can have multiple sets of perturbation
            if len(parameters) == 0:
                raise Exception('parameter not given')
            perturbation = (parameters, perturb[sim])
            primal.timeSeriesFile = primal.mesh.case + 'timeSeries_{}.txt'.format(sim)
        else:
            perturbation = None
        result = primal.run(result=initResult, startTime=startTime, dt=dts, nSteps=nSteps, 
                            writeInterval=writeInterval, reportInterval=reportInterval, 
                            mode=user.option, startIndex=startIndex, source=source, perturbation=perturbation, avgStart=avgStart)
        writeResult(user.option, [result], '{}'.format(sim), primal.timeSeriesFile)
        primal.removeStatusFile()
        # if running multiple sims reset starting index and result
        startIndex = 0
        initResult = 0.
        
