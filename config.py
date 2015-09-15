from __future__ import print_function
import time
import sys
import os
runtime = time.time()
sys.setrecursionlimit(100000)
# titan/voyager fixes
titan = 'lustre' in os.getcwd()
if titan:
    sys.path = [p for p in sys.path if 'egg' not in p]
import numpy as np
import parallel

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', action='store_true', dest='use_gpu')
parser.add_argument('--temp', action='store_true', dest='use_temp')
parser.add_argument('-p', '--no_pickle', action='store_true')
parser.add_argument('-u', '--no_unpickle', action='store_true')
parser.add_argument('-c', '--profile', action='store_true')
parser.add_argument('--profile_mem', action='store_true')
parser.add_argument('--profile_opt', action='store_true')
parser.add_argument('-s', '--python', action='store_true')
parser.add_argument('--voyager', action='store_true')
parser.add_argument('--coresPerNode', required=False, default=16, type=int)
user, args = parser.parse_known_args()

# compute type
if not user.use_gpu:
    device = 'cpu'
    precision = np.float64
else:
    device = 'gpu0'
    precision = np.float32

# theano
project = 'adFVM'
dtype = str(np.zeros(1, precision).dtype)
home = os.path.expanduser('~')
# titan check
if titan:
    home = '/lustre/atlas/proj-shared/tur103'
    assert np.__version__ == '1.9.2'
if user.use_temp:
    home = parallel.copyToTemp(home, user.coresPerNode)
#os.environ['THEANO_FLAGS'] = 'compiledir='+home+'/.theano/{0}-{1}-{2}-{3}.{4}'.format(project, device, dtype, parallel.nProcessors, parallel.rank)
os.environ['THEANO_FLAGS'] = 'lib.cnmem=0.45,compiledir='+home+'/.theano/{0}-{1}-{2}'.format(project, device, dtype)
os.environ['THEANO_FLAGS'] += ',floatX=' + dtype
os.environ['THEANO_FLAGS'] += ',device=' + device
# pickling
os.environ['THEANO_FLAGS'] += ',reoptimize_unpickled_function=False'
unpickleFunction = not user.no_unpickle
pickleFunction = not user.no_pickle
# debugging/profiling
#os.environ['THEANO_FLAGS'] += ',allow_gc=False'
#os.environ['THEANO_FLAGS'] += ',warn_float64=raise'
#os.environ['THEANO_FLAGS'] += ',nocleanup=True'
#os.environ['THEANO_FLAGS'] += ',exception_verbosity=high'
os.environ['THEANO_FLAGS'] += ',profile=' + str(user.profile)
os.environ['THEANO_FLAGS'] += ',profile_optimizer=' + str(user.profile_opt)
os.environ['THEANO_FLAGS'] += ',profile_memory=' + str(user.profile_mem)
# openmp
#os.environ['THEANO_FLAGS'] += ',openmp=True,openmp_elemwise_minsize=0'
if user.voyager:
    os.environ['THEANO_FLAGS'] += ',cxx='
    os.environ['THEANO_FLAGS'] += ',gcc.cxxflags=-I/master-usr/include/python2.7/ -I/master-usr/include/ -L/usr/lib/python2.7/config-x86_64-linux-gnu/'
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
if not user.python:
    compile_mode = 'FAST_RUN'
else:
    compile_mode = T.compile.mode.Mode(linker='py', optimizer='None')
#T.config.compute_test_value = 'raise'
#T.config.traceback.limit = -1
def inspect_inputs(i, node, fn):
    print(i, node, "input(s) value(s):", [input[0] for input in fn.inputs])
def inspect_outputs(i, node, fn):
    print("output(s) value(s):", [output[0] for output in fn.outputs])
def detect_nan(i, node, fn):
    for output in fn.outputs:
        if (isinstance(output[0],np.ndarray)):
            if (not isinstance(output[0], np.random.RandomState) and
                np.isnan(output[0]).any()):
                print('*** NaN detected ***')
                print('Inputs : %s' % [(min(input[0]), max(input[0])) for input in fn.inputs])
                print('Outputs: %s' % [(min(output[0]), max(output[0])) for output in fn.outputs])
                raise Exception('NAN')
#compile_mode = T.compile.MonitorMode(post_func=detect_nan, optimizer='None')

# LOGGING

import logging
# normal
logging.basicConfig(level=logging.WARNING)
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
cyclicPatches = ['cyclic', 'slidingPeriodic1D']
coupledPatches = cyclicPatches + processorPatches
valuePatches = processorPatches + ['calculated']
defaultPatches = coupledPatches + ['symmetryPlane', 'empty']
