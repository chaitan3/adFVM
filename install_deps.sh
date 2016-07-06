#!/bin/sh
sudo apt-get install -y build-essential 
sudo apt-get install -y python-numpy python-scipy python-matplotlib python-mpi4py 
sudo apt-get install -y python-nose python-pip

sudo pip install cython
sudo pip install theano
sudo pip install nose_parameterized

sudo apt-get install -y libmetis-dev libhdf5-openmpi-dev
#sudo pip install h5py
git clone https://github.com/h5py/h5py.git
cd h5py && \
export CC=mpicc && \
python setup.py configure --mpi && \
python setup.py build && \
sudo python setup.py install --prefix=/usr/local && \
cd .. && sudo rm -rf h5py

