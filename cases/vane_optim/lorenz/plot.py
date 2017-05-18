import numpy as np
import simulation
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
import sys

from simulation import orig_bounds
n = 48
#x = np.linspace(orig_bounds[0, 0],orig_bounds[0, 1],n)
#y = np.linspace(orig_bounds[1, 0],orig_bounds[1, 1],n)
#X, Y = np.meshgrid(x, y)
#
#evals = []
#for i in range(0, X.shape[0]):
#    evals.append([])
#    for j in range(0, len(X[i,:])):
#        evals[i].append(simulation.objective_single(((X[i,j], Y[i,j]), 0))[0])
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.scatter(X, Y, evals, c='y')
#ax.plot_surface(X, Y, evals)
#plt.show()

#np.savez('lorenz', X, Y, evals)
a=np.load('lorenz.npz')
X = a['arr_0']
Y = a['arr_1']
evals = a['arr_2']

lvls = np.logspace(-2., 1., 30)
plt.contour(X, Y, evals, norm=LogNorm(), levels=lvls)
#plt.contourf(X, Y, evals, 200)
plt.colorbar(ticks=lvls)

from pyDOE import lhs
ub = simulation.orig_bounds[:,1]
lb = simulation.orig_bounds[:,0]
n = len(simulation.default_params)
lhd = lhs(n, samples=4, criterion='maximin')
des = lb + lhd*(ub-lb)
plt.scatter(des[:, 0], des[:, 1], s=30)

s = 0
e = 30
b=np.loadtxt(sys.argv[1])
print b.shape
b = b[s:e]
x = b[:,0]
y = b[:,1]
#plt.plot(x, y, 'ko-', markerfacecolor=(1,1,1,1))
plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1)
plt.xlabel(r'$\rho$')
plt.ylabel(r'$\beta$')
plt.show()

