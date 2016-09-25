from distutils.core import setup, Extension
import numpy as np

def build_module():
    mod = Extension('function', sources=['util.c'], include_dirs=[np.get_include()])
    setup(ext_modules=[mod])

if __name__ == '__main__':
    build_module()
