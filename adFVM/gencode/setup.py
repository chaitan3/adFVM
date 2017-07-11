from distutils.core import setup, Extension
import numpy as np
import os

from os.path import expanduser
home = expanduser("~")

#os.environ['CC'] = 'ccache mpicc'
#os.environ['CXX'] = 'mpicxx'
os.environ['CC'] = 'ccache mpicc'
os.environ['CXX'] = 'mpicxx'
#os.environ['CC'] = '/home/talnikar/local/bin/gcc'
#os.environ['CXX'] = '/home/talnikar/local/bin/gcc'

incdirs = [np.get_include()]
libdirs = []
libs = []
sources = ['interface.cpp', 'density.cpp', 'adjoint.cpp', 'code.cpp']

#incdirs += [home + '/.local/include']
#libdirs += [home + '/.local/lib']
#incdirs += ['/projects/LESOpt/talnikar/local/include']
#libdirs += ['/projects/LESOpt/talnikar/local/lib/']

module = 'interface'
#compile_args = ['-std=c++11', '-O3']#, '-march=native']
compile_args = ['-std=c++11', '-O3', '-g']#, '-march=native']
mod = Extension(module,
                sources=sources,
                extra_compile_args=compile_args,
                library_dirs=libdirs,
                libraries=libs,
                include_dirs=incdirs,
                undef_macros = [ "NDEBUG" ])

setup (name = module,
       version = '0.0.1',
       description = 'This is a demo package',
       ext_modules = [mod])


