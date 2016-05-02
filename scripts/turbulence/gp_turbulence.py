import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

g = 64
n = 100
x = np.linspace(0, 1, g)
dx = dy = x[1]-x[0]
X, Y = np.meshgrid(x, x)

L = 0.02
T = 0.04
u = 4
def covariance(x1, x2):
    dist = cdist(x1, x2)
    return u**2*np.exp(-dist**2/(2*L**2))

def simulate(x):
    C = covariance(x, x) + np.eye(x.shape[0])*1e-12
    n = x.shape[0]
    z = np.random.randn(n)
    L = np.linalg.cholesky(C)
    return np.dot(L, z)

def eigen(x):
    #n = 100
    #x, w = np.polynomial.legendre.leggauss(n)
    #x = (x+1)/2
    #w /= 2.0
    m = x.shape[0]
    w = np.ones(m)*1./m

    W = np.diag(w)
    Wh = np.sqrt(W)
    C = covariance(x, x)
    Cs = np.dot(Wh, np.dot(C, Wh))

    l, v = np.linalg.eigh(Cs)
    idx = np.argsort(l)[::-1]
    l = l[idx]
    v = np.linalg.solve(Wh, v)
    v = v[:,idx]

    # truncation
    l = l[:n]
    v = v[:,:n]
    return l, v

def kl_simulate(x, l, v):
    z = np.random.rand(n)
    return (np.sqrt(l)*v*z).sum(axis=-1)

def gradient(Z):
    gz = np.zeros(Z.shape + (2,))

    gz[1:-1,:,0] = (Z[2:,:]-Z[:-2,:])/(2*dx)
    gz[0,:,0] = (Z[1,:]-Z[0,:])/dx
    gz[-1,:,0] = (Z[-1,:]-Z[-2,:])/dx

    gz[:,1:-1,1] = (Z[:,2:]-Z[:,:-2])/(2*dy)
    gz[:,0,1] = (Z[:,1]-Z[:,0])/dy
    gz[:,-1,1] = (Z[:,-1]-Z[:,-2])/dy

    return gz

x = np.vstack((X.flatten(), Y.flatten())).T

dt = 0.001
#for i in range(0, 10):
#    Z = v[:,i].reshape(X.shape)
#    plt.contourf(X, Y, Z, 50)
#    plt.colorbar()
#    plt.show()
i = 0
alpha = np.exp(-np.pi*dt/T)
print alpha
l, v = eigen(x)
print 'done'
z = np.array([0])
plt.clf()
#t = 1
t = dt
for t in np.arange(0, t, dt):
    print i
    zn = kl_simulate(x, l, v)
    zn = gradient(zn.reshape(X.shape))/50.
    if i == 0:
        z = zn.copy()
    else:
        z = (1-alpha)*zn + alpha*z
    zN = z/np.linalg.norm(z, axis=2,keepdims=1)
    Z1 = zN[:,:,0]
    Z2 = zN[:,:,1]
    Z = np.sqrt((z**2).sum(axis=2))
    vt = np.linspace(0, u/2, 101, endpoint=True)
    Z[Z > u/2] = u/2
    plt.contourf(X, Y, Z, vt)
    plt.colorbar(ticks=vt[::10])
    inter = 3
    plt.quiver(X[::inter,::inter], Y[::inter,::inter], Z1[::inter,::inter], Z2[::inter,::inter], color='k', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('images/gp_{0:04d}.png'.format(i))
    plt.clf()

    i += 1

from fft import fft
fft(Z, 'images/gp_fft.png')
#z = simulate(x)
#Z = z.reshape(X.shape)
#plt.contourf(X, Y, Z, 50)
#plt.colorbar()
#plt.show()
