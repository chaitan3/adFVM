#!/usr/bin/python2 

import h5py

files = ['mesh.hdf5', '3.hdf5']


for f in files:
    orig = h5py.File(f)
    swap = h5py.File('swapped/' + f, 'w')

    def recurse(h, h2, loc):
        for key, val in h.items():
            new_loc = loc + '/' + key
            if not isinstance(val, h5py.Dataset):
                print f, 'group', new_loc
                h2_new = h2.create_group(new_loc)
                recurse(val, h2_new, new_loc)
            else:
                print f, 'dataset', new_loc, val.dtype
                val_new = val[:].copy().byteswap().newbyteorder()
                h2.create_dataset(new_loc, data=val_new)
    recurse(orig, swap, '')
    orig.close()
    swap.close()
