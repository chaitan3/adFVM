#!/bin/bash
set -e
DIR=$(dirname $(realpath "${BASH_SOURCE[0]}"))
sudo apt-get install -y petsc-dev
sudo apt-get install -y libmetis-dev
#
sudo apt-get install -y software-properties-common
sudo sh -c "wget -O - http://dl.openfoam.org/gpg.key | apt-key add -"
sudo add-apt-repository -y http://dl.openfoam.org/ubuntu
sudo apt-get -y update
sudo apt-get -y install openfoam6

mkdir -p $DIR/sources
cd $DIR/sources

    #ar
    git clone https://github.com/RhysU/ar.git
    cd ar
        make
        python setup.py install #--user
    cd ..

    ##petsc 
    #export PETSC_DIR=~/sources/petsc-3.7.5
    #export PETSC_ARCH=arch-linux2-c-opt
    #cd petsc-3.7.5
    #    ./configure --with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90 --download-fblaslapack --download-hypre --with-debugging=0 --with-shared-libraries=1 --COPTFLAGS=-O3 --CXXOPTFLAGS=-O3 --FOPTFLAGS=-O3
    #    make
    #cd ..
    #cd petsc4py-3.7.0
    #    python setup.py build
    #    python setup.py install --prefix=~/.local

    #hdf5
    HDF5_VER=1.10.4
    wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-$HDF5_VER/src/hdf5-$HDF5_VER.tar.gz
    tar xf hdf5-.tar.gz
    cd hdf5-$HDF5_VER
    ./configure --enable-shared --enable-parallel
    make
    make install
    cd ..

    #h5py
    git clone https://github.com/h5py/h5py.git
    cd h5py 
    export CC=mpicc 
    echo $DIR/sources/hdf5-$HDF5_VER/hdf5
    python setup.py configure --mpi --hdf5=$DIR/sources/hdf5-$HDF5_VER/hdf5
    python setup.py build 
    python setup.py install #--user
    cd .. 
    

cd $DIR
