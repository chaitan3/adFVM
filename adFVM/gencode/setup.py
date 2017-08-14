from distutils.core import setup, Extension
import numpy as np
import os

from os.path import expanduser
from adFVM import cpp
cppDir = os.path.dirname(cpp.__file__)

os.environ['CC'] = 'ccache mpicc'
os.environ['CXX'] = 'mpicxx'
#os.environ['CC'] = '/home/talnikar/local/bin/gcc'
#os.environ['CXX'] = '/home/talnikar/local/bin/gcc'

home = expanduser("~")
incdirs = [np.get_include(), cppDir + '/include']
libdirs = []
libs = []
sources = ['interface.cpp', 'matop.cpp', 'mesh.cpp', 'parallel.cpp']
sources += ['kernel.cpp', 'code.cpp']

#incdirs += [home + '/.local/include']
#libdirs += [home + '/.local/lib']
#incdirs += ['/projects/LESOpt/talnikar/local/include']
#libdirs += ['/projects/LESOpt/talnikar/local/lib/']

module = 'interface'
#compile_args = ['-std=c++11', '-O3']#, '-march=native']
compile_args = ['-std=c++11', '-O3', '-g', '-march=native']
compile_args += ['-DMPI_GPU']
link_args = []
openmp = 'WITH_OPENMP' in os.environ
if openmp:
    compile_args += ['-fopenmp']
    link_args = ['-lgomp']

matop = 'WITH_MATOP' in os.environ
if matop:
    compile_args += ['-DMATOP']
    home = os.path.expanduser('~') + '/sources/petsc/'
    incdirs += [home + '/linux-gnu-c-opt/include', home + '/include']
    libdirs += [home + '/linux-gnu-c-opt/lib']
    libs += ['petsc', 'lapack']

mod = Extension(module,
                sources=sources,
                extra_compile_args=compile_args,
                extra_link_args=link_args,
                library_dirs=libdirs,
                libraries=libs,
                include_dirs=incdirs,
                undef_macros = [ "NDEBUG" ])

setup (name = module,
       version = '0.0.1',
       description = 'This is a demo package',
       ext_modules = [mod])


