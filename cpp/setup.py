from distutils.core import setup, Extension
import numpy as np
import os

os.environ['CC'] = 'ccache gcc'

mod = Extension('adFVMcpp',
                sources = ['density.cpp', 'interface.cpp', 'interp.cpp', 'op.cpp', 'timestep.cpp', 'riemann.cpp'],
                extra_compile_args=['-std=c++11', '-O3', '-march=native'],
                libraries=['adept'])
                #extra_compile_args=['-std=c++11', '-O0', '-g'])

setup (name = 'adFVMcpp',
       version = '0.0.1',
       description = 'This is a demo package',
       ext_modules = [mod],
       include_dirs=[np.get_include(), '/home/talnikar/sources/CoDiPack/include']
       )
