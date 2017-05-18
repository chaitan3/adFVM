from distutils.core import setup, Extension
import numpy as np

# define the extension module
mod = Extension('clorenz', sources=['lorenz.cpp'],include_dirs=[np.get_include()])

# run the setup
setup(ext_modules=[mod])
