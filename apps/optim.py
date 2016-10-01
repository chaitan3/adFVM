import numpy as np
import subprocess
import os, sys

from adFVM.optim import stochastic as optimizer

caseFile = sys.argv[1]
config.importModule(locals(), caseFile)
assert all(key in locals() for key in ['caseDir, genMeshParam'])

appsDir = os.path.dirname(os.path.realpath(__file__))
primal = os.path.join(appsDir, 'problem.py')
adjoint = os.path.join(appsDir, 'adjoint.py')
paramHistory = []

def readObjectiveFile(objectiveFile):
    objective = None
    gradient = []
    with open(objectiveFile, 'r') as f:
        for line in f.readlines():
            words = line.split(' ')
            if words[0] == 'objective':
                objective = float(words[-1])
            elif words[0] == 'adjoint'
                gradient.append(float(words[-1]))
    assert objective
    assert len(gradient) > 0
    return objective, gradient

def evaluate(param):
    index = len(paramHistory)
    paramHistory.append(param)
    paramDir = 'param{}'.format(index)

    genMeshParam(param, os.path.join(caseDir, paramDir)
    subprocess.call([sys.executable, primal, caseFile])

    for index in range(0, len(param)):
        perturbedParam = param.copy()
        perturbedParam[index] += 1e-6
        genMeshParam(perturbedParam, os.path.join(paramDir, 'grad{}'.format(index)))
    subprocess.call([sys.executable, adjoint, caseFile])

    return readObjectiveFile()

#optimizer
