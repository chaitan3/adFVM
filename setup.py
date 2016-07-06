import numpy as np
from distutils.core import setup, Extension
from Cython.Build import cythonize
#import setuptools

compatDir = 'adFVM/compat/'

cfuncs = cythonize(
        [Extension('cfuncs', [compatDir + 'cfuncs.pyx'], 
        libraries=['metis'], language='c++')])

setup(name='adFVM',
      version='0.1',
      description='finite volume method flow solver',
      author='Chaitanya Talnikar',
      author_email='talnikar@mit.edu',
      packages=['adFVM', 'adFVM.compat'],
      ext_modules = cfuncs,
      include_dirs=[np.get_include()]
    )
