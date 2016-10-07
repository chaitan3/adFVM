from distutils.core import setup, Extension
import numpy as np

mod = Extension('function', 
                sources=['util.c'], 
                include_dirs=[np.get_include()],
                extra_compile_args=['-g', '-O0'])
setup(ext_modules=[mod])

