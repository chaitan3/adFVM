from __future__ import print_function
import numpy as np

# switch between numpy and numpad
#import numpad as ad
#from numpad import adsparse
import numpy as ad
from scipy import sparse as adsparse
ad.value = lambda x: x

# custom norm for numpy 1.7
def norm(a, axis):
    try:
        return np.linalg.norm(a, axis=axis)
    except:
        return np.einsum('ij,ij->i', a, a)**0.5

import time

# MPI
from mpi4py import MPI
mpi = MPI.COMM_WORLD
mpi_nProcs = mpi.Get_size()
mpi_Rank = mpi.Get_rank()
mpi_processorDirectory = '/'
if mpi_nProcs > 1:
    mpi_processorDirectory = '/processor{0}/'.format(mpi_Rank)

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
        start = time.time()
        return mpi.allreduce(minData, op=MPI.MIN)
    else:
        return minData

class Exchanger(object):
    def __init__(self):
        self.requests = []

    def exchange(self, remote, sendData, recvData, tag):
        #if isinstance(sendData, ad.adarray):
        #    sendData = ad.value(sendData)
        sendRequest = mpi.Isend([sendData, MPI.DOUBLE], remote, tag)
        recvRequest = mpi.Irecv([recvData, MPI.DOUBLE], remote, tag)
        self.requests.extend([sendRequest, recvRequest])

    def wait(self):
        if mpi_nProcs == 1:
            return
        MPI.Request.Waitall(self.requests)

# LOGGING
import logging
# normal
#logging.basicConfig(level=logging.WARNING)
# debug
#logging.basicConfig(format='%(asctime)s: %(levelname)s: %(name)s: %(message)s', level=logging.INFO)
#logging.basicConfig(format='%(asctime)s: %(levelname)s: %(name)s: %(message)s', level=logging.DEBUG)

def Logger(name):
    return logging.getLogger('processor{0}:{1}'.format(mpi_Rank, name))



# CONSTANTS

SMALL = 1e-15
VSMALL = 1e-300
LARGE = 1e30

# FILE READING
import re

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
        start = data.find('(') + 1
        end = data.rfind(')')
        if fileFormat == 'binary':
            internalField = ad.array(np.fromstring(data[start:end], dtype=float))
            if vector:
                internalField = internalField.reshape((len(internalField)/3, 3))
        else:
            internalField = ad.array(np.array(extractor(data[start:end]), dtype=float))
        if not vector:
            internalField = internalField.reshape((-1, 1))
    else:
        internalField = ad.array(np.tile(np.array(extractor(data)), (size, 1)))
    return internalField

def writeField(handle, field, dtype, initial):
    handle.write(initial + ' nonuniform List<'+ dtype +'>\n')
    handle.write('{0}\n('.format(len(field)))
    if fileFormat == 'binary':
        handle.write(ad.value(field).tostring())
    else:
        handle.write('\n')
        for value in ad.value(field):
            if dtype == 'scalar':
                handle.write(str(value[0]) + '\n')
            else:
                handle.write('(' + ' '.join(np.char.mod('%f', value)) + ')\n')
    handle.write(')\n;\n')

