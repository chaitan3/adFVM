from distutils.core import setup, Extension
import numpy as np
import os
#from . import include

from os.path import expanduser
home = expanduser("~")

os.environ['CC'] = 'ccache mpicc'
os.environ['CXX'] = 'mpicxx'
#os.environ['CC'] = '/home/talnikar/local/bin/gcc'
#os.environ['CXX'] = '/home/talnikar/local/bin/gcc'

incdirs = [np.get_include()]
#incdirs += [os.path.dirname(include.__file__)]
incdirs += ['include/']
libdirs = []
libs = []
sources = ['mesh.cpp', 'cmesh.cpp']

#compile_args = ['-std=c++11', '-O3']#, '-march=native']
compile_args = ['-std=c++11', '-O3', '-g']#, '-march=native']
compile_args += ['-fopenmp']
link_args = ['-lgomp']

for module, c_args in [['cmesh', []], ['cmesh_gpu', ['-DCPU_FLOAT32']]]:
    mod = Extension(module,
                    sources=sources,
                    extra_compile_args=compile_args + c_args,
                    extra_link_args=link_args,
                    library_dirs=libdirs,
                    libraries=libs,
                    include_dirs=incdirs,
                    undef_macros = [ "NDEBUG" ])

    setup (name = module,
           version = '0.0.1',
           description = 'This is a demo package',
           ext_modules = [mod])
