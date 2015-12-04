import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

g = 50
n = 100
x = np.linspace(0, 1, g)
dx = dy = x[1]-x[0]
X, Y = np.meshgrid(x, x)

def covariance(x1, x2):
    dist = cdist(x1, x2)
    L = 0.1
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
#from scipy.fftpack import fft
#f = np.abs(fft(Z[:,10]))[:g/2]*2/g
#from scipy.signal import blackman
#w = blackman(g)
#f = np.abs(fft(w*Z[:,10]))[:g/2]*2/g
#xf = np.linspace(0.0, 1.0/(2.0*dx), g/2)
#plt.plot(xf, f)
#plt.show()
from scipy.signal import convolve2d

a = convolve2d(Z, Z, 'full')
print a.shape, a.max()
a = a[g-1:,g-1:]
print (Z*Z).sum()

plt.contourf(X, Y, Z, 50)
plt.colorbar()
plt.show()
