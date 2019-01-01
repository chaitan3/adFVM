## Introduction

adFVM is an adjoint capable unsteady compressible fluid dynamics simulation tool
for CPUs and GPUs. The flow solver uses the finite volume method (FVM) to obtain a discrete flow solution on the user provided fluid problems. The adjoint flow solution is obtained using the discrete adjoint method.
The adjoint flow solver is derived by applying automatic differentiation (provided by the library adpy) to the flow solver.

## Installation

adFVM requires the adpy library. To install it,
after cloning the main repo, execute the following
```
git submodule update --init --recursive
```
Next, follow the instructions for installing adpy mentioned
in the [adpy](https://github.com/chaitan3/adpy) repository.

adFVM requires the following packages
```
MPI: OpenMPI or similar
Python: numpy, scipy, mpi4py, cython, pytest
LAPACK
```

On Ubuntu, the following commands should be sufficient to install the above packages.
```
sudo apt install libopenmpi-dev openmpi-bin liblapack-dev
sudo apt install python-pip
pip install numpy scipy mpi4py cython
```

Build and install adFVM
```
make
python setup.py install --prefix=/path/you/want
```

Optionally, the following packages can be installed to enable
additional functionality in adFVM

1. [OpenFOAM](https://www.openfoam.com/) and [Metis](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview): Allows for the creation and modification of meshes. 
2. [h5py](https://github.com/h5py/h5py): A parallel version of h5py provides HDF5 read/write
support 
3. [ar](https://github.com/RhysU/ar): Provides statistics of long-time averaged quantities in the primal
and adjoint solvers. the library  needs to be installed.
4. [PETSc](https://www.mcs.anl.gov/petsc/): Adds artificial viscosity dissipation support
for the adjoint flow solver.

For Ubuntu, the following script can be used to install
the above packages. 
```
./install_deps.sh
```
Edit the variable PYTHONPREFIX to change the python libraries installation location.

## Testing
To run unit tests for adFVM
```
cd tests
./setup_tests.sh
./run_tests.sh
cd ..
```

## Status
[![Build Status](https://api.travis-ci.org/chaitan3/adFVM.png)](https://travis-ci.org/chaitan3/adFVM)


## Usage
To use adFVM on a flow problem, a python case
file needs to be defined that contains information about
the details of the problem, like case folder, thermodynamic constants,
design objective, simulation length, etc. Many example case files
can be found in the templates folder. For a short explanation
of the various options read the templates/vane.py file.
To run the primal flow solver, use the following
```
./apps/problem.py templates/vane.py -c
```
To run the adjoint flow solver,
```
./apps/adjoint.py templates/vane.py -c
```
The sensitivities obtained using the adjoint solver can be found
in the "objective.txt" file in the case folder. The "-c" option
is not needed if running a flow solver multiple times. 

To run on GPUs provide the "-g" option to the flow solver scripts mentioned above.
The GPU functionality is only support for Nvidia hardware and requires
the installation of the CUDA toolkit. Note that, by default the flow solver
uses 32-bit floating point precision on GPUs. In order to use 64-bit floating point 
precision on GPUs, provide the "--gpu_double" option to the flow solver scripts.

In order to run the flow solver on multiple cores, first
the mesh needs to be decomposed. Edit the "system/decomposeParDict"
file in the case folder to set the number of processors, for example 4. Then,
execute
```
decomposePar
```
To run the flow solver on 4 cores,
```
mpirun -np 4 ./apps/problem.py templates/vane.py 
```
The "objective.txt" for parallel runs is in the "processor0"
directory in the case folder.

## Contributors

Chaitanya Talnikar and Professor Qiqi Wang
