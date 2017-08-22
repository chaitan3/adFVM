from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy as np
from subprocess import check_output
import os

from adFVM import cpp

cppDir = os.path.dirname(cpp.__file__) + '/'
openmp = 'WITH_OPENMP' in os.environ
matop = 'WITH_MATOP' in os.environ
gpu = 'WITH_GPU' in os.environ
codeExt = 'cu' if gpu else 'cpp'

os.environ['CC'] = 'ccache mpicc'
#os.environ['CC'] = 'mpicc'
os.environ['CXX'] = 'mpicxx'

#def customize_compiler_for_nvcc(self):
#    '''This is a verbatim copy of the NVCC compiler extension from
#    https://github.com/rmcgibbo/npcuda-example
#    '''
#    self.src_extensions.append('.cu')
#    default_compiler_so = self.compiler_so
#    super = self._compile
#
#    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
#        if 1:#os.path.splitext(src)[1] == '.cu':
#            self.set_executable('compiler_so', 'nvcc -x cu')
#            postargs = extra_postargs + ['-Xcompiler=\"-fPIC\"', nv_arch]
#        else:
#            postargs = extra_postargs
#        super(obj, src, ext, cc_args, postargs, pp_opts)
#        self.compiler_so = default_compiler_so
#    self._compile = _compile

# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        self.compiler.src_extensions.append('.cu')
        self.compiler.set_executable('compiler_so', 'nvcc -x cu')
        self.compiler.set_executable('linker_so', 'nvcc --shared')
        #customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

home = os.path.expanduser("~")
incdirs = [np.get_include(), cppDir + '/include']
libdirs = []
libs = ['lapack']
sources = ['interface.cpp', 'mesh.cpp', 'parallel.cpp']
sources = [cppDir + x for x in sources]
sources += ['graph.cpp']
sources += [x.format(codeExt) for x in ['kernel.{}', 'code.{}']]

compile_args = ['-std=c++11', '-O3', '-g']#, '-Wall']# '-march=native']
link_args = []
if openmp:
    compile_args += ['-fopenmp']
    link_args = ['-lgomp']
if matop:
    sources += ['matop.cpp']
    compile_args += ['-DMATOP']
    home = os.path.expanduser('~') + '/sources/petsc/'
    incdirs += [home + '/linux-gnu-c-opt/include', home + '/include']
    libdirs += [home + '/linux-gnu-c-opt/lib']
    libs += ['petsc']
if gpu:
    cmdclass = {'build_ext': custom_build_ext}
    #cmdclass = {}
    nv_arch="-gencode=arch=compute_52,code=\"sm_52,compute_52\""
    compile_args += [nv_arch, '-Xcompiler=-fPIC']
    compile_args += ['-DGPU']
    mpi_incdirs = check_output('mpicc --showme | egrep -o -e "-I[a-z\/\.]*"', shell=True)
    incdirs += [d[2:] for d in mpi_incdirs.split('\n')[:-1]]
    mpi_libdirs = check_output('mpicc --showme | egrep -o -e "-L[a-z\/\.]*"', shell=True)
    libdirs += [d[2:] for d in mpi_libdirs.split('\n')[:-1]]
    libs += ['mpi']
    os.environ['CXX'] = 'nvcc --shared'
else:
    cmdclass = {}

module = 'graph'
mod = Extension(module,
                sources=sources,
                extra_compile_args=compile_args,
                extra_link_args=link_args,
                library_dirs=libdirs,
                libraries=libs,
                include_dirs=incdirs,
                undef_macros = [ "NDEBUG" ])

setup (name = module,
       version = '0.0.1',
       description = 'This is a demo package',
       ext_modules = [mod],
       cmdclass=cmdclass)


