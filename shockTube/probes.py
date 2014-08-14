#!/usr/bin/env python
from numpy import *
from matplotlib import pyplot as plt, mlab
import os

fn = 'postProcessing/sets/'
ts = array(sorted(os.listdir(fn)))
nt = len(ts)
for i in range(0, nt):
    if ts[i] == '0': continue
    data = mlab.csv2rec (fn + ts[i] + '/line_T_p_rho.xy', delimiter=' ', names=['d','T','p','rho'],converterd={'d':float, 'T':float,'p':float,'rho':float})
    print ts[i]
    plt.plot(data['d'], data['rho'])
    plt.savefig('rho' + str(i) + '.png')
    plt.clf()

