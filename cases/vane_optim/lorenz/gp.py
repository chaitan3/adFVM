import numpy as np
import scipy as sp
import scipy.linalg
from scipy.stats import norm
#from scipy.optimize import *
import nlopt
import matplotlib.pyplot as plt
from pyDOE import *

def _sanitize(x):
    if isinstance(x, list):
        x = np.array(x)
    if len(x.shape) == 1:
        return x.reshape(-1, x.shape[0])
    return x

def _optimize(fun, bounds):
    #res = differential_evolution(fun, bounds)
    #return res.x, res.fun[0]
    def nlopt_fun(x, grads):
        if grads.size > 0:
            raise Exception("!")
        else:
            return fun(x)
    opt = nlopt.opt(nlopt.GN_DIRECT_L, bounds.shape[0])
    opt.set_min_objective(nlopt_fun)
    opt.set_lower_bounds(bounds[:,0])
    opt.set_upper_bounds(bounds[:,1])
    opt.set_maxeval(1000)
    #opt.add_inequality_constraint(nlopt_constraint)
    x = (bounds[:,0] + bounds[:,1])/2
    res = opt.optimize(x)

    opt = nlopt.opt(nlopt.LN_SBPLX, bounds.shape[0])
    opt.set_min_objective(nlopt_fun)
    opt.set_lower_bounds(bounds[:,0])
    opt.set_upper_bounds(bounds[:,1])
    opt.set_maxeval(100)
    res = opt.optimize(res)

    return res, opt.last_optimum_value()

class Kernel(object):
    pass

class SquaredExponentialKernel(Kernel):
    def __init__(self, L, sigma, noise=None):
        self.L = np.array(L)
        self.sigma = sigma
        self.noise = noise

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

        x = lhs(self.ndim, samples=n, criterion='center')
        bounds = self.bounds.T
        x = bounds[[0]] + x*(bounds[[1]]-bounds[[0]])
        res = func(x)
        if len(res) > 2:
            y, yd, yn, ydn = res
            self.train(list(x), list(y), list(yd), list(yn), list(ydn))
        else:
            y, yd = res
            self.train(list(x), list(y), list(yd))

    def train(self, x, y, yd, yn=0., ydn=0.):
        if len(np.array(x).shape) == 1:
            x = [x]
            y = [y]
            yd = [yd]
            if not isinstance(yn, float):
                yn = [yn]
            if not isinstance(ydn, float):
                ydn = [ydn]
        self.x.extend(x)
        self.y.extend(y)
        self.yd.extend(yd)
        Kd = self.kernel.gradient(self.x, self.x)
        self.Kd = np.vstack((np.hstack((self.kernel.evaluate(self.x, self.x), -Kd)), np.hstack((-Kd.T, self.kernel.hessian(self.x, self.x)))))
        if self.kernel.noise is not None:
            yn, ydn = self.kernel.noise
        if isinstance(yn, float):
            yn = list(yn*np.ones_like(np.array(y)))
        if isinstance(ydn, float):
            ydn = list(ydn*np.ones_like(np.array(yd)))
        self.yn.extend(yn)
        self.ydn.extend(ydn)
        self.Kd += np.diag(np.concatenate((np.array(self.yn), np.array(self.ydn).flatten())))
        # do noise analysis.

    def posterior_min(self):
        res = _optimize(lambda x: self.evaluate(x)[0][0], self.bounds)
        print "gp min:", res
        return res[1]

    def data_min(self):
        ys = np.array(self.y).flatten()
        i = np.argmin(ys)
        print "eval min:", (self.x[i], self.y[i])
        return self.y[i]

class AcquisitionFunction(object):
    def __init__(self, gp):
        self.gp = gp

class ExpectedImprovement(AcquisitionFunction):
    def evaluate(self, xs):
        fmin = self.fmin
        mu, cov = self.gp.evaluate(xs)
        std = np.diag(cov)**0.5
        delta = fmin-mu
        Z = delta/std
        return (delta*norm.cdf(Z) + std*norm.pdf(Z))

    def optimize(self):
        self.fmin = self.gp.posterior_min()
        res = _optimize(lambda x: -self.evaluate(x)[0], self.gp.bounds)
        return res[0]

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
