#!/usr/bin/env python
from bcs import g
from matplotlib import pyplot as plt, mlab
from matplotlib import markers as mk
import os
from profile import get_length, pressure, suction, c
from numpy import *


fn = 'postProcessing/surfaces/'
ts = os.listdir(fn)
nt = len(ts)

def get_data(patch, i):
    #p0 = [196359.330811, 166833.772321]
    p0 = [171372.]
    f = fn + ts[i] + '/p_' + patch + '.raw'
    data = mlab.csv2rec (f, delimiter=' ', names=['x','y', 'z', 'p'],converterd={'x':float, 'y':float,'z':float,'p':float})
    p = minimum(data['p'], p0[i])
    Ma = (sqrt(2.0/(g-1)*((p0[i]/p)**((g-1)/g)-1)))

    return Ma, data['x']

im = plt.imread('/home/talnikar/Research/pics/blade/blade-velocity.png')
plt.imshow(im)

def transx(a):
    return 77 + (684-77)*(a/1.4)
def transy(a):
    return 461 - (461-32)*(a/2.0)

for i in range(0, nt):
    Map, coordsp = get_data('pressure', i)
    Mas, coordss = get_data('suction', i)
    print(Map)
    print(Mas)
    sp = get_length(pressure, coordsp)
    ss = get_length(suction, coordss)

    fill = 0.5
    if i == 1:
        fill=1

    plt.scatter(transx(sp/c), transy(Map), c='r', s=10, alpha=fill)
    plt.scatter(transx(ss/c), transy(Mas), c='b', s=10, alpha=fill)

plt.show()
