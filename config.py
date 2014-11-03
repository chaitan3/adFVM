from __future__ import print_function

# switch between numpy and numpad
#import numpad as ad
#from numpad import adsparse
import numpy as ad
from scipy import sparse as adsparse
ad.value = lambda x: x

# custom norm for numpy 1.7
import numpy as np
def norm(a, axis):
    try:
        return np.linalg.norm(a, axis=axis)
    except:
        return np.einsum('ij,ij->i', a, a)**0.5

# LOGGING
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

fileFormat = 'binary'
#fileFormat = 'ascii'
foamFile = {'version':'2.0', 'format': fileFormat, 'class': 'volScalarField', 'object': ''}

# group patches
processorPatches = ['processor', 'processorCyclic']
coupledPatches = ['cyclic', 'processor', 'processorCyclic']
valuePatches = ['processor', 'processorCyclic', 'calculated']
defaultPatches = ['cyclic', 'symmetryPlane', 'empty', 'processor', 'processorCyclic']

