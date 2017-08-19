from distutils.core import setup, Extension
import numpy as np
import os
#from . import include

from os.path import expanduser
home = expanduser("~")

#os.environ['CC'] = '/home/talnikar/local/bin/gcc'
#os.environ['CXX'] = '/home/talnikar/local/bin/gcc'
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
