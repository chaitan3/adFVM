from distutils.core import setup, Extension
import numpy as np
import mpi4py
import os

from os.path import expanduser
home = expanduser("~")

os.environ['CC'] = 'ccache mpicc'
#os.environ['CC'] = 'mpicc'
os.environ['CXX'] = 'mpicxx'
#os.environ['LDSHARED'] = 'mpicc'

for module, args in zip(['adFVMcpp', 'adFVMcpp_ad'], ['', '-DADIFF']):
    compile_args = ['-std=c++11', '-O3', '-march=native']
    if len(args) > 0:
        compile_args += [args]
    mod = Extension(module,
                    sources = ['density.cpp', 'interface.cpp', 'interp.cpp', 'op.cpp', 'timestep.cpp', 'riemann.cpp', 'objective.cpp'],
                    extra_compile_args=compile_args,
                    #extra_linker_args=['--whole-archive'],
                    libraries=['AMPI'])
                    #extra_compile_args=['-std=c++11', '-O0', '-g'])

    setup (name = module,
           version = '0.0.1',
           description = 'This is a demo package',
           ext_modules = [mod],
           include_dirs=[np.get_include(), home + '/sources/CoDiPack/include', '/usr/local/include']
           )
