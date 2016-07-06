import numpy as np
import os
from setuptools import setup, Extension, Command
from Cython.Build import cythonize

class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('rm -vrf ./build ./dist ./*.so ./*.pyc ./*.tgz ./*.egg-info')


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
      ],
      cmdclass={
        'clean': CleanCommand,
      }
    )
