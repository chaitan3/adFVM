#!/usr/bin/python2

import numpy as np
import subprocess
import os, sys, glob, shutil

from adFVM import config

caseFile = sys.argv[1]
config.importModule(locals(), caseFile)
#assert all(key in locals() for key in ['caseDir, genMeshParam, nParam'])

appsDir = os.path.dirname(os.path.realpath(__file__))
primal = os.path.join(appsDir, 'problem.py')
adjoint = os.path.join(appsDir, 'adjoint.py')
paramHistory = []
eps = 1e-6

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

def evaluate(param):
    index = len(paramHistory)
    paramHistory.append(param)
    paramDir = os.path.join(caseDir, 'param{}'.format(index))

    if not os.path.exists(paramDir):
        os.makedirs(paramDir)
    for pkl in glob.glob(os.path.join(caseDir, '*.pkl')):
        shutil.copy(pkl, paramDir)
    for hdf in glob.glob(os.path.join(caseDir,'*.hdf5')):
        shutil.copy(pkl, paramDir)
    
    shutil.copy(caseFile, paramDir)
    problemFile = os.path.join(paramDir, os.path.basename(caseFile))
    with open(problemFile, 'r') as f:
        lines = f.readlines()
    with open(problemFile, 'w') as f:
        for line in lines:
            writeLine = line.replace('CASEDIR', '\'{}\''.format(paramDir))
            f.write(writeLine)

    genMeshParam(param, paramDir)
    subprocess.call([sys.executable, primal, problemFile])

    for index in range(0, len(param)):
        perturbedParam = param.copy()
        perturbedParam[index] += eps
        genMeshParam(perturbedParam, os.path.join(paramDir, 'grad{}'.format(index)))
    subprocess.call([sys.executable, adjoint, problemFile])

    return readObjectiveFile(os.path.join(paramDir, 'objective.txt'))

#from adFVM.optim import stochastic as optimizer
evaluate([1.0])
