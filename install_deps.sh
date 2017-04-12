#!/bin/sh
sudo apt-get install -y build-essential 
sudo apt-get install -y python-numpy python-scipy python-matplotlib python-mpi4py 
sudo apt-get install -y python-nose python-pip
sudo apt-get install -y libmetis-dev 

pip install cython --user

sudo apt-get install -y python-h5py
#sudo apt-get install -y libhdf5-openmpi-dev
#git clone https://github.com/h5py/h5py.git
#cd h5py && \
#export CC=mpicc && \
#python setup.py configure --mpi && \
#python setup.py build && \
#python setup.py install --prefix=~/.local && \
#cd .. && rm -rf h5py
#

sudo apt-get install -y ccache
sudo apt-get install -y petsc-dev
