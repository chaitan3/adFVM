## Introduction

adFVM is an explicit unsteady compressible fluid dynamics package with adjoint capability
using automatic differentiation provided by the library adpy.

## Installation
adFVM requires the adpy library. To install it,
after cloning, execute the following
```
git submodule update --init --recursive
```
Next build and install it using the following (using python2)
```
cd adpy
python setup.py build
python setup.py install --prefix=/path/you/want
cd ..
```

adFVM requires the following python packages, they
will be automatically installed when you execute the python install script or
you can install them separately if you want.
```
numpy
scipy
mpi4py
cython
matplotlib
```
Build and install adFVM
```
python setup.py build
python setup.py install --prefix=/path/you/want
```

Additionally for HDF5 capability parallel h5py needs to be
installed. For running the adjoint solver using artificial
viscosity PETSc and lapack need to be installed. For computing
statistics of long-time averaged quantities in the primal
and adjoint solvers the library [ar](https://github.com/RhysU/ar) needs to be installed.
For handling meshes OpenFOAM and Metis need to be installed.
For Ubuntu, the following script can be used to install
all these packages. 
```
./install_deps.sh
```
Edit the variable PYTHONPREFIX to change the python libraries installation location.

## Testing
To run unit tests for adpy
```
cd adpy/tests
./run_tests.sh
cd ../..
```
To run unit tests for adFVM
```
cd tests
./run_tests.sh
cd ..
```

## Usage

## Contributors

Chaitanya Talnikar and Professor Qiqi Wang
