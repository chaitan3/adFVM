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

def obj(u, r=0.):
    return u[:,2]**2/(2*N)

def dJdu_obj(u, r=0., mod=np):
    dJdu = mod.zeros(u.shape)
    dJdu[:,2] = u[:,2]/N
    return dJdu

dt = 0.001
Ni = 50000
x = np.array([1, 0, -1])
for i in range(0, Ni):
    x = x + lorenz(x, 28)*dt
#N = 10000
N = 1000
xs = [x]
for i in range(0, N):
    x = xs[-1]
    xn = x + lorenz(x, 28)*dt
    xs.append(xn)
xs = np.array(xs)
#plt.plot(xs)
#plt.show()

ti = np.random.randn(3,1)
ts = [ti.flatten()]
for i in range(0, N):
    x, t = xs[i], ts[-1]
    dfdu = dfdu_lorenz(x, 28).reshape(3,3)
    tn = t + np.dot(dfdu, t)*dt
    ts.append(tn.flatten())
ts = np.array(ts)
#plt.plot(ts)
#plt.show()
    
n = N + 1

def dot(x, y):
    return (x*y).sum(axis=-1)

lamb = 1e1
def f(u):
    x, p = u[:3*n], u[3*n:6*n]
    #lambc = lamb
    v = u[6*n:]
    v = v.reshape((N, 1))
    lambc = lamb + v
    x = x.reshape((n, 3))
    p = p.reshape((n, 3))

    dfdu = dfdu_lorenz(ad.array(xs[:-1]), 28, mod=ad)*dt + ad.eye(3).reshape((1,3,3))
    dJdu = dJdu_obj(ad.array(xs[:-1]), mod=ad)
    #dJdu = 0.
    fx = x[1:]+p[1:]*dot(x[:-1], x[:-1]).reshape((-1,1))/lambc - dot(dfdu, x[:-1].reshape((N, 1, 3))) - dJdu
    fp = p[:-1]+x[:-1]*dot(p[1:], p[1:]).reshape((-1,1))/lambc - x[:-1] - dot(dfdu.transpose((0, 2, 1)), p[1:].reshape((N, 1, 3)))

    #return ad.concatenate((x[0]-ad.array(ti.flatten()), ad.ravel(fx), ad.ravel(fp), p[-1]-x[-1]))
    ##constraint
    #cons = 1e-2
    cons = 1e-2
    frob = dot(x[:-1], x[:-1])*dot(p[1:], p[1:])/(ad.ravel(lamb)**2)
    return ad.concatenate((x[0]-ad.array(ti.flatten()), ad.ravel(fx), ad.ravel(fp), p[-1]-x[-1], ad.ravel(v)*(frob-cons)))

xi = ts.flatten()
pi = xi
#u0 = np.concatenate((xi, pi))
v = np.zeros(N)
u0 = np.concatenate((xi, pi, v))

u0 = ad.array(u0)
u = ad.solve(f, u0, max_iter=1000)

x, p = u[:3*n]._value, u[3*n:6*n]._value
x = x.reshape(n, 3)
p = p.reshape(n, 3)
#L = p[1:].reshape(N, 1, 3)*x[:-1].reshape(N, 3, 1)/lamb
v = u[6*n:]._value
L = p[1:].reshape(N, 1, 3)*x[:-1].reshape(N, 3, 1)/(lamb + v.reshape(N, 1, 1))**2
plt.plot(x)
#plt.plot(v)
plt.show()
#plt.plot(p)
#plt.plot(dot(x[:-1], x[:-1])*dot(p[1:], p[1:])/(100.+v)**2)
plt.plot(L[:,:,0])
plt.show()
