#!/bin/sh
set -e
PWD=`pwd`
sudo apt-get install -y petsc-dev liblapack-dev
sudo apt-get install -y python-h5py
sudo apt-get install -y libhdf5-openmpi-dev

sudo apt-get install -y libmetis-dev
sudo sh -c "wget -O - http://dl.openfoam.org/gpg.key | apt-key add -"
sudo add-apt-repository http://dl.openfoam.org/ubuntu
sudo apt-get -y update
sudo apt-get -y install openfoam5

mkdir -p ~/sources
cd ~/sources
    #ar
    git clone https://github.com/RhysU/ar.git
    cd ar
        make
        python setup.py install --prefix=~/.local 
    cd ..

    #petsc 
    #export PETSC_DIR=~/sources/petsc-3.7.5
    #export PETSC_ARCH=arch-linux2-c-opt
    #cd petsc-3.7.5
    #    ./configure --with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90 --download-fblaslapack --download-hypre --with-debugging=0 --with-shared-libraries=1 --COPTFLAGS=-O3 --CXXOPTFLAGS=-O3 --FOPTFLAGS=-O3
    #    make
    #cd ..
    #cd petsc4py-3.7.0
    #    python setup.py build
    #    python setup.py install --prefix=~/.local

    # h5py
    #git clone https://github.com/h5py/h5py.git
    #cd h5py && \
    #export CC=mpicc && \
    #python setup.py configure --mpi && \
    #python setup.py build && \
    #python setup.py install --prefix=~/.local && \
    #cd .. && rm -rf h5py
    #
cd $PWD
