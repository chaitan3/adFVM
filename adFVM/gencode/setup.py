from distutils.core import setup, Extension
import numpy as np
import os

from os.path import expanduser
home = expanduser("~")

os.environ['CC'] = 'ccache mpicc'
os.environ['CXX'] = 'mpicxx'
#os.environ['CC'] = '/home/talnikar/local/bin/gcc'
#os.environ['CXX'] = '/home/talnikar/local/bin/gcc'

incdirs = [np.get_include()]
libdirs = []
libs = []
sources = ['interface.cpp', 'matop.cpp', 'kernel.cpp', 'code.cpp']

#incdirs += [home + '/.local/include']
#libdirs += [home + '/.local/lib']
#incdirs += ['/projects/LESOpt/talnikar/local/include']
#libdirs += ['/projects/LESOpt/talnikar/local/lib/']

module = 'interface'
#compile_args = ['-std=c++11', '-O3']#, '-march=native']
compile_args = ['-std=c++11', '-O3', '-g']#, '-march=native']

matop = True
matop = False
if matop:
    compile_args += ['-DMATOP']
    home = os.path.expanduser('~') + '/sources/petsc/'
    incdirs += [home + '/linux-gnu-c-opt/include', home + '/include']
    libdirs += [home + '/linux-gnu-c-opt/lib']
    libs += ['petsc', 'lapack']

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


