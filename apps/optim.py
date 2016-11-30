#!/usr/bin/python2

import numpy as np
import os, sys, glob, shutil
import subprocess

from adFVM import config

caseFile = sys.argv[1]
config.importModule(locals(), caseFile)
#assert all(key in locals() for key in ['caseDir, genMeshParam, nParam, spawnJob, paramBounds'])

appsDir = os.path.dirname(os.path.realpath(__file__))
primal = os.path.join(appsDir, 'problem.py')
adjoint = os.path.join(appsDir, 'adjoint.py')
paramHistory = []
eps = 1e-5

def readObjectiveFile(objectiveFile):
    objective = None
    gradient = []
    with open(objectiveFile, 'r') as f:
        for line in f.readlines(): 
            words = line.split(' ')
            if words[0] == 'objective':
                objective = float(words[-1])
            elif words[0] == 'adjoint':
                gradient.append(float(words[-1])/eps)
    assert objective
    assert len(gradient) > 0
    return objective, gradient

def evaluate(param, genAdjoint=True, runSimulation=True):
    index = len(paramHistory)
    paramHistory.append(param)
    paramDir = os.path.join(caseDir, 'param{}'.format(index))

    os.makedirs(paramDir)
    for pkl in glob.glob(os.path.join(caseDir, '*.pkl')):
        shutil.copy(pkl, paramDir)
    for hdf in glob.glob(os.path.join(caseDir,'*.hdf5')):
        shutil.copy(hdf, paramDir)
    
    shutil.copy(caseFile, paramDir)
    problemFile = os.path.join(paramDir, os.path.basename(caseFile))
    with open(problemFile, 'r') as f:
        lines = f.readlines()
    with open(problemFile, 'w') as f:
        for line in lines:
            writeLine = line.replace('CASEDIR', '\'{}\''.format(paramDir))
            f.write(writeLine)

    try:
        genMeshParam(param, paramDir)
    except (OSError, subprocess.CalledProcessError) as e:
        print('Gen primal mesh param failed')
        raise
    return

    if genAdjoint:
        for index in range(0, len(param)):
            perturbedParam = param.copy()
            perturbedParam[index] += eps
            gradDir = os.path.join(paramDir, 'grad{}'.format(index))
            os.makedirs(gradDir)
            try:
                genMeshParam(perturbedParam, gradDir)
            except (OSError, subprocess.CalledProcessError) as e:
                print('Gen adjoint mesh param failed')
                raise

    if runSimulation:
        spawnJob([sys.executable, primal, problemFile])
        spawnJob([sys.executable, adjoint, problemFile])

        return readObjectiveFile(os.path.join(paramDir, 'objective.txt'))
    return

from adFVM.optim import designOfExperiment
print designOfExperiment(lambda x: evaluate(x, False, False), paramBounds, 2*nParam)
#print evaluate(np.zeros(8)*1., False, False)

