import os
import sys
import glob
import shutil

h = sys.argv[1]
f = sys.argv[2]

fs = glob.glob(h + '/{}.*.png'.format(f))
fs.sort()

fsr = fs[::-1]

for x1, x2 in zip(fs, fsr):
    shutil.copyfile(x2, h + '/rev_{}'.format(os.path.basename(x1)))

