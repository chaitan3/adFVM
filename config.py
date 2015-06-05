from __future__ import print_function
import sys
#sys.path = [p for p in sys.path if '1.9.2' not in p]
import numpy as np
import parallel

# compute type
device = 'cpu'
precision = np.float64
#device = 'gpu0'
#precision = np.float32

# theano
import os
sys.setrecursionlimit(100000)
project = 'adFVM'
dtype = str(np.zeros(1, precision).dtype)
home= os.path.expanduser('~')
#home = '/lustre/atlas/proj-shared/tur103'
#assert np.__version__ == '1.7.1'
# whether to copy theano dir to temp
#home = parallel.copyToTemp(home)
#os.environ['THEANO_FLAGS'] = 'compiledir='+home+'/.theano/{0}-{1}-{2}-{3}.{4}'.format(project, device, dtype, parallel.nProcessors, parallel.rank)
os.environ['THEANO_FLAGS'] = 'compiledir='+home+'/.theano/{0}-{1}-{2}'.format(project, device, dtype)
os.environ['THEANO_FLAGS'] += ',floatX=' + dtype
os.environ['THEANO_FLAGS'] += ',device=' + device
# pickling
os.environ['THEANO_FLAGS'] += ',reoptimize_unpickled_function=False'
unpickleFunction = True
pickleFunction = True
# profiling, gc, cleanup
#os.environ['THEANO_FLAGS'] += ',allow_gc=False'
#os.environ['THEANO_FLAGS'] += ',nocleanup=True'
#os.environ['THEANO_FLAGS'] += ',exception_verbosity=high'
#os.environ['THEANO_FLAGS'] += ',profile=True'
#os.environ['THEANO_FLAGS'] += ',profile_optimizer=True'
#os.environ['THEANO_FLAGS'] += ',profile_memory=True'
# openmp
#os.environ['THEANO_FLAGS'] += ',openmp=True,openmp_elemwise_minsize=0'
# for voyager
#os.environ['THEANO_FLAGS'] += ',cxx='
#os.environ['THEANO_FLAGS'] += ',gcc.cxxflags=-I/master-usr/include/python2.7/ -I/master-usr/include/ -L/usr/lib/python2.7/config-x86_64-linux-gnu/'
import theano as T
import theano.tensor as ad
import theano.sparse as adsparse
ad.array = lambda x: x
ad.value = lambda x: x
broadcastPattern = (False, True)
ad.bcmatrix = ad.TensorType(dtype, broadcastable=broadcastPattern)
def bcalloc(value, shape):
    X = ad.alloc(value, *shape)
    if shape[1:] == (1,):
        X = ad.patternbroadcast(X, broadcastPattern)
    return X
ad.bcalloc = bcalloc
# debugging/compiling
#compile_mode = 'FAST_COMPILE'
compile_mode = 'FAST_RUN'
#compile_mode = T.compile.mode.Mode(linker='py', optimizer='None')
#T.config.compute_test_value = 'raise'
def inspect_inputs(i, node, fn):
    print(i, node, "input(s) value(s):", [input[0] for input in fn.inputs])
def inspect_outputs(i, node, fn):
    print("output(s) value(s):", [output[0] for output in fn.outputs])

# custom norm for numpy 1.7
def norm(a, axis, **kwargs):
    try:
        #return np.linalg.norm(a, axis=axis, keepdims=True)
        return np.linalg.norm(a, axis=axis).reshape((-1,1))
    except:
        return np.sqrt(np.einsum('ij,ij->i', a, a)).reshape((-1,1))

# LOGGING

import logging
# normal
#logging.basicConfig(level=logging.WARNING)
# debug
#logging.basicConfig(format='%(asctime)s: %(levelname)s: %(name)s: %(message)s', level=logging.INFO)
#logging.basicConfig(format='%(asctime)s: %(levelname)s: %(name)s: %(message)s', level=logging.DEBUG)
def Logger(name):
    return logging.getLogger('processor{0}:{1}'.format(parallel.rank, name))

# CONSTANTS
if precision == np.float64:
    SMALL = 1e-15
    VSMALL = 1e-300
    LARGE = 1e300
else:
    SMALL = 1e-9
    VSMALL = 1e-30
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
