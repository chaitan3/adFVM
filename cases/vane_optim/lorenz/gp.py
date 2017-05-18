import numpy as np
import scipy as sp
import scipy.linalg
from scipy.stats import norm
from scipy.optimize import *
import matplotlib.pyplot as plt

def _sanitize(x):
    if isinstance(x, list):
        x = np.array(x)
    if len(x.shape) == 1:
        return x.reshape(-1, x.shape[0])
    return x

class Kernel(object):
    pass

class SquaredExponentialKernel(Kernel):
    def __init__(self, L, sigma):
        self.L = np.array(L)
        self.sigma = sigma

    def evaluate(self, x, xp):
        L, sigma = self.L, self.sigma
        L = L[None,None,:]
        x = _sanitize(x)
        xp = _sanitize(xp)
        d = (x[:,None,:]-xp[None,:,:])/L
        return sigma**2*np.exp(-(d**2).sum(axis=-1)/2)

    def gradient(self, x, xp):
        L, sigma = self.L, self.sigma
        L = L[None,None,:]
        x = _sanitize(x)
        xp = _sanitize(xp)
        d = x[:,None,:]-xp[None,:,:]
        K = -self.evaluate(x, xp)[:,:,None]*d/(L**2)
        K = K.reshape(K.shape[0], K.shape[1]*K.shape[2])
        return K

    def hessian(self, x, xp):
        L, sigma = self.L, self.sigma
        L = L[None,None,:]
        x = _sanitize(x)
        xp = _sanitize(xp)
        n = x.shape[-1]
        d = (x[:,None,:]-xp[None,:,:])/L**2
        I = np.eye(n)[None, None, :, :]
        K = (I/L[:,:,:,None]**2-d[:,:,:,None]*d[:,:,None,:])*self.evaluate(x, xp)[:,:,None,None]
        K = K.transpose((0, 2, 1, 3)).reshape(K.shape[0]*K.shape[2], K.shape[1]*K.shape[3])
        return K

class GaussianProcess(object):
    def __init__(self, kernel, bounds):
        self.kernel = kernel
        self.x = []
        self.y = []
        self.yd = []
        self.yn = []
        self.ydn = []
        self.bounds = np.array(bounds)
        self.ndim = self.bounds.shape[0]
    
    def evaluate(self, xs):
        xs = _sanitize(xs)
        x, y, yd = np.array(self.x), np.array(self.y), np.array(self.yd)
        d = np.concatenate((y, yd.flatten()))
        Ki = np.hstack((self.kernel.evaluate(xs, x), -self.kernel.gradient(xs, x)))

        K = self.kernel.evaluate(xs, xs)
        Kd = self.Kd
        Kd = Kd + 1e-12*np.diag(np.ones_like(np.diag(Kd)))
        L = sp.linalg.cho_factor(Kd)
        mu = np.dot(Ki, sp.linalg.cho_solve(L, d))
        cov = K - np.dot(Ki, sp.linalg.cho_solve(L, Ki.T))
        return mu, cov

    def explore(self, n, func):
        assert len(self.x) == 0

        x = np.random.rand(self.ndim, n)
        x = self.bounds[:,[0]]+ x*(self.bounds[:,[1]]-self.bounds[:,[0]])
        x = x.T
        y, yd = func(x)
        self.train(list(x), list(y), list(yd))

    def train(self, x, y, yd, yn=0., ydn=0.):
        if len(np.array(x).shape) == 1:
            x = [x]
            y = [y]
            yd = [yd]
        self.x.extend(x)
        self.y.extend(y)
        self.yd.extend(yd)
        Kd = self.kernel.gradient(self.x, self.x)
        self.Kd = np.vstack((np.hstack((self.kernel.evaluate(self.x, self.x), -Kd)), np.hstack((-Kd.T, self.kernel.hessian(self.x, self.x)))))
        if isinstance(yn, float):
            yn = list(yn*np.ones_like(np.array(y)))
        if isinstance(ydn, float):
            ydn = list(ydn*np.ones_like(np.array(yd)))
        self.yn.extend(yn)
        self.ydn.extend(ydn)
        self.Kd += np.diag(np.concatenate((np.array(self.yn), np.array(self.ydn).flatten())))
        # do noise analysis.

class AcquisitionFunction(object):
    def __init__(self, gp):
        self.gp = gp

class ExpectedImprovement(AcquisitionFunction):
    def evaluate(self, xs):
        fmin = np.min(self.gp.y)
        mu, cov = self.gp.evaluate(xs)
        std = np.diag(cov)**0.5
        delta = fmin-mu
        Z = delta/std
        return (delta*norm.cdf(Z) + std*norm.pdf(Z))[0]

    def optimize(self):
        res = differential_evolution(lambda x: -self.evaluate(x), self.gp.bounds)
        xd = res.x
        return xd

def test_func(x):
    x = _sanitize(x)
    return np.sin(x).reshape((x.shape[0])),\
           np.cos(x)

def _test_main():
    kernel = SquaredExponentialKernel([3.], 1.)
    bounds = [[0, 4*2*np.pi]]
    gp = GaussianProcess(kernel, bounds)
    gp.explore(3, test_func)
    xs = np.linspace(gp.bounds[0,0], gp.bounds[0,1], 500).reshape(-1,1)

    ei = ExpectedImprovement(gp)
    for i in range(0, 100):
        x = ei.optimize()
        y, yd = test_func(x)
        gp.train(x, y, yd)

        plt.ylim([-2,2])
        plt.scatter(gp.x, gp.y, c='k')
        #plt.plot(xs, expected_improvement(xs))

        mu, cov = gp.evaluate(xs)
        std = np.diag(cov)**0.5
        plt.plot(xs.flatten(), mu)
        plt.fill_between(xs.flatten(), mu-std, mu + std, facecolor='gray')

        plt.show()

if __name__ == "__main__":
    _test_main()
