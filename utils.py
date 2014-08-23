from __future__ import print_function

# LOGGING
import logging
# normal
#logging.basicConfig(level=logging.WARNING)
# debug
#logging.basicConfig(format='%(asctime)s: %(levelname)s: %(name)s: %(message)s', level=logging.INFO)
#logging.basicConfig(format='%(asctime)s: %(levelname)s: %(name)s: %(message)s', level=logging.DEBUG)

def logger(name):
    return logging.getLogger(name)

# MPI
from mpi4py import MPI
mpi = MPI.COMM_WORLD
mpi_nProcs = mpi.Get_size()
mpi_Rank = mpi.Get_rank()
mpi_processorDirectory = ''
if mpi_nProcs > 1:
    mpi_processorDirectory = 'processor{0}'.format(mpi_Rank)

def pprint(*args, **kwargs):
    if mpi_Rank == 0:
        print(*args, **kwargs)
def max(data):
    maxData = np.max(data)
    if mpi_nProcs > 1:
        return mpi.allreduce(maxData, op=MPI.MAX)
    else:
        return maxData

def min(data):
    minData = np.min(data)
    if mpi_nProcs > 1:
        return mpi.allreduce(minData, op=MPI.MIN)
    else:
        return minData

# CONSTANTS

SMALL = 1e-15
VSMALL = 1e-300
LARGE = 1e30

# FILE READING
import re
import numpy as np
import numpad as ad

foamHeader = '''/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  2.2.2                                 |
|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
'''
foamFile = {'version':'2.0', 'format': 'ascii', 'class': 'volScalarField', 'object': ''}

def removeCruft(content, keepHeader=False):
    # remove comments and newlines
    content = re.sub(re.compile('/\*.*\*/',re.DOTALL ) , '' , content)
    content = re.sub(re.compile('//.*\n' ) , '' , content)
    content = re.sub(re.compile('\n\n' ) , '\n' , content)
    # remove header
    if not keepHeader:
        content = re.sub(re.compile('FoamFile\n{(.*?)}\n', re.DOTALL), '', content)
    return content

def extractField(data, size, vector):
    extractScalar = lambda x: re.findall('[0-9\.Ee\-]+', x)
    if vector:
        extractor = lambda y: list(map(extractScalar, re.findall('\(([0-9\.Ee\-\r\n\s\t]+)\)', y)))
    else:
        extractor = extractScalar
    nonUniform = re.search('nonuniform', data)
    data = re.search(re.compile('[A-Za-z<>\s\r\n]+(.*)', re.DOTALL), data).group(1)
    if nonUniform is not None:
        start = data.find('(')
        internalField = ad.adarray(extractor(data[start:]))
        if not vector:
            internalField = internalField.reshape((-1, 1))
    else:
        internalField = ad.adarray(np.tile(np.array(extractor(data)), (size, 1)))
    return internalField


