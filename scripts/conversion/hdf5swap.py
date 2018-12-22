#!/usr/bin/python 

import os
import sys
import h5py
import shutil

files = sys.argv[1:]

for f in files:
    origf = f[:-5] + '_orig.hdf5'
    shutil.move(f, origf)
    orig = h5py.File(origf)
    swap = h5py.File(f, 'w')

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
    os.remove(origf)
