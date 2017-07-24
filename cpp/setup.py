from distutils.core import setup, Extension
import numpy as np
import mpi4py
import os

from os.path import expanduser
home = expanduser("~")
#matop = True
matop = False

#os.environ['CC'] = 'ccache mpicc'
#os.environ['CXX'] = 'mpicxx'
os.environ['CC'] = 'mpicc'
os.environ['CXX'] = 'mpicxx'
#os.environ['CC'] = '/home/talnikar/local/bin/gcc'
#os.environ['CXX'] = '/home/talnikar/local/bin/gcc'

incdirs = [np.get_include(), home + '/sources/CoDiPack/include']
libdirs = []
libs = []
libs += ['AMPI'] 
sources = ['density.cpp', 'interface.cpp', 'interp.cpp', 'op.cpp', 'timestep.cpp', 'riemann.cpp', 'objective.cpp']
if matop:
    sources += ['matop.cpp']

incdirs += [home + '/.local/include']
libdirs += [home + '/.local/lib']
incdirs += ['/projects/LESOpt/talnikar/local/include']
libdirs += ['/projects/LESOpt/talnikar/local/lib/']

if matop:
#petscdir = '/usr/lib/petscdir/3.6.2/x86_64-linux-gnu-real/'
    petscdir = home + '/sources/petsc/linux-gnu-c-opt'
    incdirs += [petscdir + '/../include', petscdir + '/include']
    libdirs += [petscdir + '/lib']
    libs += ['petsc']
    #libs += ['super']

adjargs = '-DADIFF'
if matop:
    adjargs += ' -DMATOP'

for module, args in zip(['adFVMcpp', 'adFVMcpp_ad'], ['', adjargs]):
#for module, args in zip(['adFVMcpp'], ['']):
#for module, args in zip(['adFVMcpp_ad'], [adjargs]):
    compile_args = ['-std=c++11', '-O3']#, '-march=native']
    #compile_args=['-std=c++11', '-O0', '-g']
    if len(args) > 0:
        compile_args += [args]
    mod = Extension(module,
                    sources=sources,
                    extra_compile_args=compile_args,
                    library_dirs=libdirs,
                    libraries=libs,
                    include_dirs=incdirs)

    setup (name = module,
           version = '0.0.1',
           description = 'This is a demo package',
           ext_modules = [mod])


