# Copyright Qiqi Wang (qiqi@mit.edu) 2013

import sys
#from pylab import *
#from numpy import *
import numpy as np
import matplotlib.pyplot as plt

import numpad as ad

sys.path.append('..')
#from lssode import *
from scipy.optimize import root

#set_fd_step(1E-30j)
sigma, beta = 10, 8./3

def lorenz(u, rho):
    shp = u.shape
    x, y, z = u.reshape([-1, 3]).T
    dxdt, dydt, dzdt = sigma*(y-x), x*(rho-z)-y, x*y - beta*z
    return np.transpose([dxdt, dydt, dzdt]).reshape(shp)

def dfdu_lorenz(u, rho, mod=np, per=0.0):
    shp = u.shape
    x, y, z = u.reshape([-1, 3]).T
    x = x.reshape((-1,1,1))
    y = x.reshape((-1,1,1))
    z = x.reshape((-1,1,1))
    one = mod.ones(x.shape)
    f1 = mod.hstack((-sigma*one - per, sigma*one, 0*one))
    f2 = mod.hstack((rho-z, -one - per, -x))
    f3 = mod.hstack((y, x, -beta*one - per))
    dfdu = mod.concatenate((f1, f2, f3), axis=2)
    return dfdu

dt = 0.001
Ni = 50000
x = np.array([1, 0, -1])
for i in range(0, Ni):
    x = x + lorenz(x, 28)*dt
#N = 10000
N = 10000
xs = [x]
for i in range(0, N):
    x = xs[-1]
    xn = x + lorenz(x, 28)*dt
    xs.append(xn)
#plot(xs)
#show()

ti = np.random.randn(3,1)
ts = [ti.flatten()]
for i in range(0, N):
    x, t = xs[i], ts[-1]
    dfdu = dfdu_lorenz(x, 28).reshape(3,3)
    tn = t + np.dot(dfdu, t)*dt
    ts.append(tn.flatten())
#plot(ts)
#show()
    
n = N + 1
def f(u):
    x, p = u[:3*n], u[3*n:]
    x = x.reshape((n, 3))
    p = p.reshape((n, 3))
    def dot(x, y):
        return (x*y).sum(axis=-1)
    dfdu = dfdu_lorenz(x[:-1], 28, mod=ad)*dt + ad.eye(3).reshape((1,3,3))
    lamb = 100.
    fx = x[1:]+1./lamb*p[1:]*(dot(x[:-1], x[:-1]).reshape((-1,1))) - dot(dfdu, x[:-1].reshape((N, 1, 3)))
    fp = p[:-1]+1./lamb*x[:-1]*(dot(p[1:], p[1:]).reshape((-1,1))) - x[:-1] - dot(dfdu.transpose((0, 2, 1)), p[1:].reshape((N, 1, 3)))
    return ad.concatenate((x[0]-ad.array(ti.flatten()), ad.ravel(fx), ad.ravel(fp), p[-1]-x[-1]))

u0 = ad.array(np.random.randn(6*n))
u = ad.solve(f, u0, max_iter=1000)

x, p = u[:3*n]._value, u[3*n:]._value
x = x.reshape(n, 3)
p = p.reshape(n, 3)
L = p.reshape(n, 1, 3)*x.reshape(n, 3, 1)
plt.plot(x)
plt.show()
