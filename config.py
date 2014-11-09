from __future__ import print_function
import numpy as np

import theano.tensor as ad
import theano.sparse as adsparse
import theano as T
ad.array = lambda x: x
ad.value = lambda x: x

from theano.ifelse import ifelse
def smin(a, b):
    return ifelse(ad.lt(a, b), a, b)

# custom norm for numpy 1.7
def norm(a, axis):
    try:
        return np.linalg.norm(a, axis=axis)
    except:
        return np.einsum('ij,ij->i', a, a)**0.5

# LOGGING
#T.config.compute_test_value = 'raise'
def inspect_inputs(i, node, fn):
    print(i, node, "input(s) value(s):", [input[0] for input in fn.inputs])

def inspect_outputs(i, node, fn):
    print("output(s) value(s):", [output[0] for output in fn.outputs])
import logging
# normal
#logging.basicConfig(level=logging.WARNING)
# debug
#logging.basicConfig(format='%(asctime)s: %(levelname)s: %(name)s: %(message)s', level=logging.INFO)
#logging.basicConfig(format='%(asctime)s: %(levelname)s: %(name)s: %(message)s', level=logging.DEBUG)
import parallel
def Logger(name):
    return logging.getLogger('processor{0}:{1}'.format(parallel.rank, name))

# CONSTANTS
SMALL = 1e-15
VSMALL = 1e-300
LARGE = 1e30

# FILE READING

foamHeader = '''/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  2.2.2                                 |
|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
'''

#fileFormat = 'binary'
fileFormat = 'ascii'
foamFile = {'version':'2.0', 'format': fileFormat, 'class': 'volScalarField', 'object': ''}

# group patches
processorPatches = ['processor', 'processorCyclic']
coupledPatches = ['cyclic', 'processor', 'processorCyclic']
valuePatches = ['processor', 'processorCyclic', 'calculated']
defaultPatches = ['cyclic', 'symmetryPlane', 'empty', 'processor', 'processorCyclic']

