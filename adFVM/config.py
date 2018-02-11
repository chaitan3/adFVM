from __future__ import division
from __future__ import print_function
# mira hack
#import tensorflow as tf

import time
runtime = time.time()

import os
import sys
import subprocess
sys.setrecursionlimit(100000)
py3 = not (sys.version_info[0] < 3)
sys.path.insert(0, '/master/home/talnikar/.local/lib/python2.7/site-packages')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', action='store_true', dest='use_gpu')
parser.add_argument('--gpu_double', action='store_true', dest='use_gpu_double')
parser.add_argument('-m', '--omp', action='store_true', dest='use_openmp')
parser.add_argument('-p', '--matop', action='store_true', dest='use_matop')
parser.add_argument('-d', '--hdf5', action='store_true')
parser.add_argument('-o', '--profile', action='store_true', dest='profile')
parser.add_argument('-k', '--gc', action='store_true', dest='use_gc')
parser.add_argument('--temp', action='store_true', dest='use_temp')

parser.add_argument('-c', '--compile', action='store_true')
parser.add_argument('-e', '--compile_exit', action='store_true')
parser.add_argument('-l', '--no_compile', action='store_true')

user, args = parser.parse_known_args()

import numpy as np

np.random.seed(3)
from . import parallel
#parallel.pprint('Running on {} threads'.format(user.coresPerNode))

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
gpu = user.use_gpu
gpu_double = user.use_gpu_double

if user.use_gpu:
    import ctypes
    ctypes.CDLL('libgomp.so.1', mode=ctypes.RTLD_GLOBAL)
    if user.use_gpu_double:
        precision = np.float64
    else:
        precision = np.float32
    codeExt = 'cu'
else:
    precision = np.float64
    codeExt = 'cpp'
openmp = user.use_openmp
gc = user.use_gc
profile = user.profile
compile = (user.compile or user.compile_exit) and (parallel.rank == 0)

def get_compiler_args():
    compiler = 'ccache mpicc'
    linker = 'mpicxx'
    libs = []
    incdirs = []
    libdirs = []
    sources = []
    extra_compile_args = []

    cppDir = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'cpp')
    incdirs += [os.path.join(cppDir, 'include')]
    sources += [os.path.join(cppDir, x) for x in ['external.cpp', 'mesh.cpp', 'parallel.cpp', 'scaling.cpp']]

    if gpu:
        mpi_incdirs = subprocess.check_output('mpicc --showme | egrep -o -e "-I[a-z\/\.]*"', shell=True)
        incdirs += [d[2:] for d in mpi_incdirs.split('\n')[:-1]]
        mpi_libdirs = subprocess.check_output('mpicc --showme | egrep -o -e "-L[a-z\/\.]*"', shell=True)
        libdirs += [d[2:] for d in mpi_libdirs.split('\n')[:-1]]
        libs += ['mpi', 'cublas', 'cusolver']
    else:
        libs += ['lapack']
    if matop:
        sources += [os.path.join(cppDir, 'matop.cpp')]
        extra_compile_args += ['-DMATOP']
        if gpu and not gpu_double:
            extra_compile_args += ['-DPETSC_USE_REAL_SINGLE']
            home = os.path.expanduser('~') + '/sources/petsc_single/'
        else:
            home = os.path.expanduser('~') + '/sources/petsc/'
        build = 'arch-linux2-c-opt'
        incdirs += ['{}/{}/include'.format(home, build), home + '/include', os.path.expanduser('~') + '/sources/cusp/']
        libdirs += ['{}/{}/lib'.format(home, build)]
        libs += ['petsc']

    return {'compiler': compiler,
            'linker': linker,
            'libs': libs,
            'incdirs': incdirs,
            'libdirs': libdirs,
            'sources': sources,
            'extra_compile_args': extra_compile_args}

import adpy.config
adpy.config.set_config(sys.modules[__name__])

matop = user.use_matop
hdf5 = user.hdf5
compile_exit = user.compile_exit

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

fileFormat = 'binary'
#fileFormat = 'ascii'
foamFile = {'version':'2.0', 'format': fileFormat, 'class': 'volScalarField', 'object': ''}

# group patches
processorPatches = ['processor', 'processorCyclic']
cyclicPatches = ['cyclic', 'slidingPeriodic1D']
coupledPatches = cyclicPatches + processorPatches
defaultPatches = coupledPatches + ['symmetryPlane', 'empty']
#defaultPatches = coupledPatches + ['empty']

