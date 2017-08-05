import numpy as np
import os
import sys
sys.path.insert(0, '/master/home/talnikar/.local/lib/python2.7/site-packages/')
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
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')


compatDir = 'adFVM/compat/'
cfuncs = cythonize(
        [Extension(compatDir + 'cfuncs', [compatDir + 'cfuncs.pyx'], 
        language='c++')])

setup(name='adFVM',
      version='0.1.1',
      description='finite volume method flow solver',
      author='Chaitanya Talnikar',
      author_email='talnikar@mit.edu',
      packages=['adFVM', 'adFVM.compat'],
      package_data={
      'adFVM': ['gencode/*.cpp', 'gencode/*.hpp', 'gencode/Makefile', 'gencode/setup.py'],
      },
      include_package_data=True,
      ext_modules = cfuncs,
      include_dirs=[np.get_include()],
      install_requires=[ 
          'numpy >= 1.8.2',
          'scipy >= 0.13.3',
          'mpi4py >= 0.13.1',
          'cython >= 0.24',
          #'h5py >= 2.6.0',
          #'theano >= 0.7.0',
          'matplotlib >= 1.3.1'
      ],
      cmdclass={
        'clean': CleanCommand,
      }
    )
