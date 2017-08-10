from __future__ import division
from __future__ import print_function
# mira hack
#import tensorflow as tf

import time
runtime = time.time()

import os
import sys
sys.setrecursionlimit(100000)
py3 = not (sys.version_info[0] < 3)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', action='store_true', dest='use_gpu')
parser.add_argument('--temp', action='store_true', dest='use_temp')
parser.add_argument('-p', '--no_pickle', action='store_true')
parser.add_argument('-u', '--no_unpickle', action='store_true')
parser.add_argument('-o', '--profile', action='store_true')
parser.add_argument('-c', '--compile', action='store_true')
parser.add_argument('-l', '--no_compile', action='store_true')
parser.add_argument('--profile_mem', action='store_true')
parser.add_argument('--profile_opt', action='store_true')
parser.add_argument('-s', '--python', action='store_true')
parser.add_argument('--voyager', action='store_true')
parser.add_argument('--titan', action='store_true')
parser.add_argument('--bw', action='store_true')
parser.add_argument('--mira', action='store_true')
parser.add_argument('-d', '--hdf5', action='store_true')
parser.add_argument('--coresPerNode', required=False, default=0, type=int)
user, args = parser.parse_known_args()

if user.titan:
    sys.path = [p for p in sys.path if 'egg' not in p]

import numpy as np
np.random.seed(3)
from . import parallel
parallel.pprint('Running on {} threads'.format(user.coresPerNode))

def exceptInfo(e, info=''):
    rank = parallel.rank
    #raise type(e), \
    #      type(e)(e.message + ' '), \
    #      sys.exc_info()[2]
    e.args += (info, rank)
    raise 

def stop():
    import pdb; pdb.set_trace()

# compute type
if not user.use_gpu:
    gpu = False
    precision = np.float64
    #precision = np.float32
else:
    gpu = True
    precision = np.float32

compile = user.compile and (parallel.rank == 0)

#import tensorflow as ad
#ad.sum = ad.reduce_sum
#import tensorflow as adsparse
#if not user.use_gpu:
#    dtype = ad.float64
#    ad.device('/cpu:0')
#else:
#    dtype = ad.float32
#    ad.device('/gpu:0')
#
#def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
#    # Need to generate a unique name to avoid duplicates:
#    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
#
#    ad.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
#    g = ad.get_default_graph()
#    with g.gradient_override_map({"PyFunc": rnd_name}):
#        return ad.py_func(func, inp, Tout, stateful=stateful, name=name)

# theano
#project = 'adFVM'
#dtype = str(np.zeros(1, precision).dtype)
## titan check
#if user.titan:
#    home = '/lustre/atlas/proj-shared/tur103'
#    assert np.__version__ == '1.9.2'
#elif user.bw:
#    home = '/scratch/sciteam/talnikar/stable/'
#elif user.mira:
#    home = '/projects/LESOpt/talnikar/local/'
#else:
#    home = os.path.expanduser('~')
#if user.use_temp:
#    home = parallel.copyToTemp(home, user.coresPerNode)
##os.environ['THEANO_FLAGS'] = 'compiledir='+home+'/.theano/{0}-{1}-{2}-{3}.{4}'.format(project, device, dtype, parallel.nProcessors, parallel.rank)
#os.environ['THEANO_FLAGS'] = 'lib.cnmem=0.45'
#os.environ['THEANO_FLAGS'] += ',base_compiledir='+home+'/.theano'
#os.environ['THEANO_FLAGS'] += ',compiledir='+home+'/.theano/{0}-{1}-{2}'.format(project, device, dtype)
#os.environ['THEANO_FLAGS'] += ',floatX=' + dtype
#os.environ['THEANO_FLAGS'] += ',device=' + device
## pickling
#os.environ['THEANO_FLAGS'] += ',reoptimize_unpickled_function=False'
#compile = user.compile
#unpickleFunction = not user.no_unpickle
#pickleFunction = not user.no_pickle
## debugging/profiling
##os.environ['THEANO_FLAGS'] += ',allow_gc=False'
##os.environ['THEANO_FLAGS'] += ',warn_float64=raise'
##os.environ['THEANO_FLAGS'] += ',nocleanup=True'
##os.environ['THEANO_FLAGS'] += ',exception_verbosity=high'
#os.environ['THEANO_FLAGS'] += ',profile=' + str(user.profile)
#if user.profile:
#    os.environ['THEANO_FLAGS'] += ',profiling.destination=profile_{}_{}.out'.format(device, parallel.rank)
#os.environ['THEANO_FLAGS'] += ',profile_optimizer=' + str(user.profile_opt)
#os.environ['THEANO_FLAGS'] += ',profile_memory=' + str(user.profile_mem)
## openmp
##os.environ['THEANO_FLAGS'] += ',openmp=True,openmp_elemwise_minsize=0'
#if user.voyager:
#    os.environ['THEANO_FLAGS'] += ',gcc.cxxflags=-I/master-usr/include/python2.7/ -I/master-usr/include/ -L/usr/lib/python2.7/config-x86_64-linux-gnu/'
#elif user.bw:
#    os.environ['THEANO_FLAGS'] += ',gcc.cxxflags=-march=native'
#elif user.mira:
#    os.environ['THEANO_FLAGS'] += ',gcc.cxxflags=-L'+home+'lib'
#if user.no_compile:
#    # needs modification to theano/gof/cmodule.py version_str
#    # and maybe gcc.cxxflags
#    os.environ['THEANO_FLAGS'] += ',cxx='
#    #os.environ['THEANO_FLAGS'] += ',gcc.cxxflags=-march=native'
#elif user.mira:
#    os.environ['THEANO_FLAGS'] += ',cxx=powerpc64-bgq-linux-g++'
#
#import theano as T
#import theano.tensor as ad
#import theano.sparse as adsparse
#from theano.ifelse import ifelse
#ad.ifelse = ifelse
#broadcastPattern = (False, True)
#ad.bcmatrix = ad.TensorType(dtype, broadcastable=broadcastPattern)
#ad.bctensor3 = ad.TensorType(dtype, broadcastable=(False,False,True))
#def bcalloc(value, shape):
#    X = ad.alloc(value, *shape)
#    if shape[1:] == (1,):
#        X = ad.patternbroadcast(X, broadcastPattern)
#    return X
#ad.bcalloc = bcalloc
## debugging/compiling
#if not user.python:
#    compile_mode = 'FAST_RUN'
#    #compile_mode = 'FAST_COMPILE'
#else:
#    compile_mode = T.compile.mode.Mode(linker='py', optimizer='None')
##T.config.compute_test_value = 'raise'
##T.config.traceback.limit = -1
#def inspect_inputs(i, node, fn):
#    print(i, node, "input(s) value(s):", [input[0] for input in fn.inputs])
#def inspect_outputs(i, node, fn):
#    print("output(s) value(s):", [output[0] for output in fn.outputs])
#def detect_nan(i, node, fn):
#    for output in fn.outputs:
#        if (isinstance(output[0],np.ndarray)):
#            if (not isinstance(output[0], np.random.RandomState) and
#                np.isnan(output[0]).any()):
#                print('*** NaN detected ***')
#                print('Inputs : %s' % [(min(input[0]), max(input[0])) for input in fn.inputs])
#                print('Outputs: %s' % [(min(output[0]), max(output[0])) for output in fn.outputs])
#                raise Exception('NAN')
##compile_mode = T.compile.MonitorMode(post_func=detect_nan, optimizer='None')

# LOGGING

import logging
# normal
logging.basicConfig(level=logging.WARNING)
# debug
#logging.basicConfig(format='%(asctime)s: %(levelname)s: %(name)s: %(message)s', level=logging.INFO)
#logging.basicConfig(format='%(asctime)s: %(levelname)s: %(name)s: %(message)s', level=logging.DEBUG)
def Logger(name):
    return logging.getLogger('processor{0}:{1}'.format(parallel.rank, name))

from contextlib import contextmanager
@contextmanager
def suppressOutput():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
def importModule(localsDict, module):
    caseDir, caseFile = os.path.split(module)
    sys.path.append(os.path.abspath(caseDir))
    caseFile = __import__(caseFile.split('.')[0])
    for attr in dir(caseFile):
        if not attr.startswith('_'):
            # defines primal, objective and perturb, nSteps, writeInterval, startTime, dt
            localsDict[attr] = getattr(caseFile, attr)
    return 

def timeFunction(string):
    def decorator(function):
        def wrapper(*args, **kwargs):
            start = time.time()
            output = function(*args, **kwargs)
            parallel.mpi.Barrier()
            end = time.time()
            parallel.pprint(string + ':', end-start)
            return output
        return wrapper
    return decorator

# CONSTANTS
if precision == np.float64:
    SMALL = 1e-30
    VSMALL = 1e-300
    LARGE = 1e300
else:
    SMALL = 1e-9
    VSMALL = 1e-30
    LARGE = 1e30

def isfloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# FILE READING

foamHeader = '''/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  2.2.2                                 |
|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
'''
hdf5 = user.hdf5
#hdf5 = True

fileFormat = 'binary'
#fileFormat = 'ascii'
foamFile = {'version':'2.0', 'format': fileFormat, 'class': 'volScalarField', 'object': ''}

# group patches
processorPatches = ['processor', 'processorCyclic']
cyclicPatches = ['cyclic', 'slidingPeriodic1D']
coupledPatches = cyclicPatches + processorPatches
defaultPatches = coupledPatches + ['symmetryPlane', 'empty']
#defaultPatches = coupledPatches + ['empty']
