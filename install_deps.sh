#!/bin/sh
sudo apt-get install build-essential 
sudo apt-get install python-numpy python-scipy python-matplotlib python-mpi4py
sudo pip install cython

sudo pip install theano
sudo pip install nose_parameterized

sudo apt-get install libmetis-dev libhdf5-openmpi-dev
#sudo pip install h5py
git clone https://github.com/h5py/h5py.git
cd h5py && \
python setup.py configure --mpi && \
python setup.py build && \
sudo python setup.py install --prefix=/usr/local
rm -rf h5py

