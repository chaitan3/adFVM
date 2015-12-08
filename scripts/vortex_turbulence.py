import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

g = 64
n = 100
x = np.linspace(0, 1, g)
dx = dy = x[1]-x[0]
X, Y = np.meshgrid(x, x)
S = 1

L = 0.02
T = 0.1
N = 1000
dt = 0.1
u = L/T
k = np.random.randn(N, 3)/2
k = np.linalg.norm(k, axis=1)
L = L/k
print L
#wrms = np.sqrt(eps/3*nu)
wrms = 10.

# characteristic time dropoff

def random_unit_vec():
    a = np.random.rand()*2*np.pi
    return np.array([np.cos(a), np.sin(a)])

x = np.vstack((X.flatten(), Y.flatten())).T
xis = []
for i in range(0, N):
    xi = np.random.rand(2)
    xis.append(xi)
for t in np.arange(0, 1, dt):
    z = 0
    for i in range(0, N):
        xis[i] += u*random_unit_vec()*dt
        xi = xis[i]
        # random walk on xi
        xd = x - xi
        xdn2 = -(xd**2).sum(axis=1).reshape(-1,1)
        #xdr = np.zeros_like(xd)
        a = np.random.rand()*2*np.pi
        xdr = np.zeros_like(xd)
        xdr[:,0] = np.sin(a)*xd[:,1]
        xdr[:,1] = -np.sin(a)*xd[:,0]
        xe = np.exp(xdn2/(2*L[i]**2))
        C = np.sqrt(12*np.pi*S*L[i]**2/N)*wrms**2
        z += -C/(2*np.pi)*(1-xe)*xe*xdr/xdn2
    # rotation?
    Z1 = z[:,0].reshape(X.shape)
    Z2 = z[:,1].reshape(X.shape)
    Z = (z**2).sum(axis=1).reshape(X.shape)
    plt.contourf(X, Y, Z, 50)
    plt.colorbar()
    inter = 3
    plt.quiver(X[::inter,::inter], Y[::inter,::inter], Z1[::inter,::inter], Z2[::inter,::inter], color='k', linewidth=2)
    plt.show()

# temporal correlation
# * new realization and correlate
# * langevin equation
# * divergence free constraint using internal velocity
