#!/usr/bin/env python
from bcs import T01, hflux, Twall
from matplotlib import pyplot as plt, mlab
from matplotlib import markers as mk
import os
from profile import get_length, pressure, suction
from numpy import *


fn = 'postProcessing/surfaces/'
ts = os.listdir(fn)
nt = len(ts)

def get_data(patch, i):
    #f = fn + ts[i] + '/T_' + patch + '.raw'
    f = fn + ts[i] + '/wallHeatFlux_' + patch + '.raw'
    data = mlab.csv2rec (f, delimiter=' ', names=['x','y', 'z', 'h'],converterd={'x':float, 'y':float,'z':float,'h':float})
    h = -data['h']/(T01-Twall)
    #h = hflux/(T01-data['h'])
    h = minimum(maximum(0, h), 1600)

    return h, data['x']

im = plt.imread('../../pics/blade/blade-htc1.png')
plt.imshow(im)

def transx(a):
    return 43 + (373-43)*((a+100)/200)
def transy(a):
    return 283 - (283-15)*(a/1600)

hp, coordsp = get_data('pressure', 0)
hs, coordss = get_data('suction', 0)
sp = get_length(pressure, coordsp)
ss = get_length(suction, coordss)

fill = 1

plt.scatter(transx(-sp*1000), transy(hp), c='r', s=10, alpha=fill,marker='+')
plt.scatter(transx(ss*1000), transy(hs), c='b', s=10, alpha=fill, marker='+')

plt.show()
