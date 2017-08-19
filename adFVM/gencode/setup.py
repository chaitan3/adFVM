from distutils.core import setup, Extension
import numpy as np
import os

from adFVM import cpp

cppDir = os.path.dirname(cpp.__file__) + '/'
openmp = 'WITH_OPENMP' in os.environ
matop = 'WITH_MATOP' in os.environ
gpu = 'WITH_GPU' in os.environ
codeExt = 'cu' if gpu else 'cpp'

os.environ['CC'] = 'ccache mpicc'
os.environ['CXX'] = 'mpicxx'
#os.environ['CC'] = '/home/talnikar/local/bin/gcc'
#os.environ['CXX'] = '/home/talnikar/local/bin/gcc'

home = os.path.expanduser("~")
incdirs = [np.get_include(), cppDir + '/include']
libdirs = []
libs = ['lapack']
sources = ['interface.cpp', 'mesh.cpp', 'parallel.cpp']
sources = [cppDir + x for x in sources]
sources += ['graph.cpp']
sources += [x.format(codeExt) for x in ['kernel.{}', 'code.{}']]

#incdirs += [home + '/.local/include']
#libdirs += [home + '/.local/lib']
#incdirs += ['/projects/LESOpt/talnikar/local/include']
#libdirs += ['/projects/LESOpt/talnikar/local/lib/']

module = 'graph'
#compile_args = ['-std=c++11', '-O3']#, '-march=native']

compile_args = ['-std=c++11', '-O3', '-g', '-march=native']
link_args = []
if openmp:
    compile_args += ['-fopenmp']
    link_args = ['-lgomp']

if matop:
    sources += ['matop.cpp']
    compile_args += ['-DMATOP']
    home = os.path.expanduser('~') + '/sources/petsc/'
    incdirs += [home + '/linux-gnu-c-opt/include', home + '/include']
    libdirs += [home + '/linux-gnu-c-opt/lib']
    libs += ['petsc']

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


