import h5py
import sys

hdf5File = h5py.File(sys.argv[1], 'r+')
boundary = hdf5File['/U/boundary']

def equal(a, b):
    for m, n in zip(a, b):
        if m != n:
            return False
    return True

old = ['left', 'type', 'calculated']
new = ['left', 'type', 'zeroGradient']

for index, data in enumerate(boundary):
    if equal(data, old):
        boundary[index] = new

hdf5File.close()
        
