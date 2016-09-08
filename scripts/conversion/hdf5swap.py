#!/usr/bin/python2 

import h5py

files = ['mesh.hdf5', '3.hdf5']


for f in files:
    orig = h5py.File(f)
    #swap = h5py.File('swapped/' + f)

    def recurse(h, loc):
        for key in h.keys()
            val = h[key]
            if isinstance(val, h5py.Dataset):
                print 'group', key, loc
                recurse(key, loc + '/' + key)
            else:
                print 'val', key, val.dtype, loc
    recurse(orig, '/')
