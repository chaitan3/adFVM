import numpy as np
from distutils.core import setup, Extension
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("cfuncs", ["cfuncs.pyx"])],
    include_dirs=[np.get_include()]
)

setup(
    ext_modules = [Extension("cext", ["cext.c", "part_mesh.c"],
        extra_link_args=['-lmetis'])],
    include_dirs=[np.get_include()]
)
