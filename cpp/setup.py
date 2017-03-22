from distutils.core import setup, Extension
import numpy as np
import mpi4py
import os

from os.path import expanduser
home = expanduser("~")

os.environ['CC'] = 'ccache mpicc'
os.environ['CXX'] = 'mpicxx'
#os.environ['CC'] = 'mpicc'
#os.environ['CXX'] = 'mpicxx'
#os.environ['CC'] = '/home/talnikar/local/bin/gcc'
#os.environ['CXX'] = '/home/talnikar/local/bin/gcc'

incdirs = [np.get_include(), home + '/sources/CoDiPack/include']
libdirs = []
#incdirs += ['/projects/LESOpt/talnikar/local/include']
#libdirs += ['/projects/LESOpt/talnikar/local/lib/']

for module, args in zip(['adFVMcpp', 'adFVMcpp_ad'], ['', '-DADIFF']):
    compile_args = ['-std=c++11', '-O3']#, '-march=native']
    #compile_args=['-std=c++11', '-O0', '-g']
    if len(args) > 0:
        compile_args += [args]
    mod = Extension(module,
                    sources = ['density.cpp', 'interface.cpp', 'interp.cpp', 'op.cpp', 'timestep.cpp', 'riemann.cpp', 'objective.cpp'],
                    extra_compile_args=compile_args,
                    library_dirs=libdirs,
                    libraries=['AMPI'],
                    include_dirs=incdirs)

    setup (name = module,
           version = '0.0.1',
           description = 'This is a demo package',
           ext_modules = [mod])

