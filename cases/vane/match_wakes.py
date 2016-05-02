#!/usr/bin/env python
from bcs import g, p0, speed
from matplotlib import pyplot as plt, mlab
from matplotlib import markers as mk
import os
from numpy import *

fn = 'postProcessing/surfaces/'
ts = os.listdir(fn)
nt = len(ts)

def get_file(t, i):
    f = fn + ts[i] + '/' + t + '_downstream.raw'
    return mlab.csv2rec (f, delimiter=' ', names=['x','y', 'z', t],converterd={'x':float, 'y':float,'z':float,t:float})

def get_data(i):
    p = get_file('p', i)
    T = get_file('T', i)
    u = get_file('magU', i)
    dpt = p0-p['p']*(1+(g-1)*(u['magU']/speed(T['T']))**2/2)**(g/(g-1))

    return dpt/133.322368, p['y']

#im = plt.imread('../../pics/blade/blade-wakes.png')
#plt.imshow(im)

def transx(a):
    return 77 + (684-77)*(a/1.4)
def transy(a):
    return 461 - (461-32)*(a/2.0)

for i in range(0, nt):
    y, x = get_data(i)
    fill = 0.5
    plt.plot(x, y, 'r+')
    #plt.scatter(transx(x), transy(y), c='r', s=10, alpha=fill)

plt.show()
