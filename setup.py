import numpy as np
#from distutils.core import setup, Extension
from setuptools import setup, Extension
from Cython.Build import cythonize

compatDir = 'adFVM/compat/'

cfuncs = cythonize(
        [Extension(compatDir + 'cfuncs', [compatDir + 'cfuncs.pyx'], 
        libraries=['metis'], language='c++')])

setup(name='adFVM',
      version='0.1',
      description='finite volume method flow solver',
      author='Chaitanya Talnikar',
      author_email='talnikar@mit.edu',
      packages=['adFVM', 'adFVM.compat'],
      ext_modules = cfuncs,
      include_dirs=[np.get_include()],
      install_requires=[ 
          'numpy >= 1.8.2',
          'scipy >= 0.13.3',
          'mpi4py >= 0.13.1',
          'cython >= 0.24',
          'theano',
          'h5py >= 2.6.0',
          'matplotlib'
      ]
    )
