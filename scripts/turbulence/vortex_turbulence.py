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
N = 1000
dt = 0.001
k = np.random.randn(N, 3)/2
k = np.linalg.norm(k, axis=1)
L = L/k
#wrms = np.sqrt(eps/3*nu)
u = 4.
wrms = 6.

T = 10*L/u
# characteristic time dropoff

def random_unit_vec():
    a = np.random.rand()*2*np.pi
    return np.array([np.cos(a), np.sin(a)])

x = np.vstack((X.flatten(), Y.flatten())).T
xis = []
ais = []
tis = []
for i in range(0, N):
    xi = np.random.rand(2)
    xis.append(xi)
    a = np.random.rand()*2*np.pi
    ais.append(a)   
    ti = np.random.rand()*T[i]
    tis.append(ti)
    
j = 0
#t = 1
t = dt
for t in np.arange(0, t, dt):
    z = 0
    print j
    for i in range(0, N):
        if tis[i] > T[i]:
            xis[i] = np.random.rand(2)
            tis[i] = 0.
        else:
            xis[i] += u*random_unit_vec()*dt
        xi = xis[i]
            
        # random walk on xi
        xd = x - xi
        xdn2 = -(xd**2).sum(axis=1).reshape(-1,1)
        #xdr = np.zeros_like(xd)
        a = ais[i]
        xdr = np.zeros_like(xd)
        xdr[:,0] = np.sin(a)*xd[:,1]
        xdr[:,1] = -np.sin(a)*xd[:,0]
        xe = np.exp(xdn2/(2*L[i]**2))
        tf = tis[i]/T[i]
        C = 4*tf*(1-tf)*np.sqrt(12*np.pi*S*L[i]**2/N)*wrms**2
        z += -C/(2*np.pi)*(1-xe)*xe*xdr/xdn2
        tis[i] += dt
    # rotation?
    Z = (z**2).sum(axis=1).reshape(X.shape)
    zN = z/np.linalg.norm(z, axis=1).reshape(-1,1)
    Z1 = zN[:,0].reshape(X.shape)
    Z2 = zN[:,1].reshape(X.shape)
    v = np.linspace(0, u/2, 101, endpoint=True)
    Z[Z > u/2] = u/2
    plt.contourf(X, Y, Z, v)
    plt.colorbar(ticks=v[::10])
    inter = 3
    plt.quiver(X[::inter,::inter], Y[::inter,::inter], Z1[::inter,::inter], Z2[::inter,::inter], scale=30, color='k', linewidth=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('images/vortex_{0:04d}.png'.format(j))
    plt.clf()
    j += 1

from fft import fft
fft(z[:,1].reshape(X.shape), 'images/vortex_fft.png')


