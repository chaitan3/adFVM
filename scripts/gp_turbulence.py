import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

g = 64
n = 100
x = np.linspace(0, 1, g)
dx = dy = x[1]-x[0]
X, Y = np.meshgrid(x, x)

def covariance(x1, x2):
    dist = cdist(x1, x2)
    L = 0.04
    return np.exp(-dist**2/(2*L**2))

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

    gz[1:-1,:,0] = (Z[1:,:]-Z[:-1,:])/(2*dx)
    gz[0,:,0] = (Z[1,:]-Z[0,:])/dx
    gz[-1,:,0] = (Z[-1,:]-Z[-2,:])/dx

    gz[:,1:-1,1] = (Z[:,1:]-Z[:,:-1])/(2*dy)
    gz[:,0,1] = (Z[:,1]-Z[:,0])/dy
    gz[:,-1,1] = (Z[:,-1]-Z[:,-2])/dy

    return gz

def corr2d(Z):
    m, n = Z.shape
    res = np.zeros_like(Z)
    for i in range(0, m):
        x = m-i
        for j in range(0, n):
            y = n-j
            res[i, j] = (Z[:x,:y]*Z[i:,j:]).sum()/(x*y)
    return res

x = np.vstack((X.flatten(), Y.flatten())).T

#l, v = eigen(x)
##for i in range(0, 10):
##    Z = v[:,i].reshape(X.shape)
##    plt.contourf(X, Y, Z, 50)
##    plt.colorbar()
##    plt.show()
#z = kl_simulate(x, l, v)
#Z = z.reshape(X.shape)
#plt.contourf(X, Y, Z, 50)
#plt.colorbar()
#plt.show()

z = simulate(x)
Z = z.reshape(X.shape)

#from scipy.fftpack import fft2, fftshift
#Z = corr2d(Z)
#f = fftshift(fft2(Z))
##f = f[:g/2,:g/2]
#f = np.abs(f)**2
#plt.imshow(np.log(f))

plt.contourf(X, Y, Z, 50)
plt.colorbar()
plt.show()
