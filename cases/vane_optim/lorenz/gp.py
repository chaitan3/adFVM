import numpy as np
import scipy as sp
import scipy.linalg
from scipy.stats import norm
from scipy.optimize import *
import nlopt
from pyDOE import *
import gp_noder

beta = 0.

def _sanitize(x):
    if isinstance(x, list):
        x = np.array(x)
    if len(x.shape) == 1:
        return x.reshape(-1, x.shape[0])
    return x

def _optimize(fun, bounds, cons=None):
    #res = differential_evolution(fun, bounds)
    #return res.x, res.fun[0]
    def nlopt_fun(x, grads):
        if grads.size > 0:
            raise Exception("!")
        else:
            if cons:
                return fun(x) + 100*max(0, cons(x))
            else:
                return fun(x)

    def nlopt_constraint(x, grads):
        if grads.size > 0:
            raise Exception("!")
        else:
            return 0.5-cons(x)

    opt = nlopt.opt(nlopt.GN_DIRECT_L, bounds.shape[0])
    opt.set_min_objective(nlopt_fun)
    opt.set_lower_bounds(bounds[:,0])
    opt.set_upper_bounds(bounds[:,1])
    opt.set_maxeval(1000)
    #if cons:
    #    opt.add_inequality_constraint(nlopt_constraint)
    x = (bounds[:,0] + bounds[:,1])/2
    res = opt.optimize(x)

    opt = nlopt.opt(nlopt.LN_SBPLX, bounds.shape[0])
    #opt = nlopt.opt(nlopt.LN_COBYLA, bounds.shape[0])
    opt.set_min_objective(nlopt_fun)
    opt.set_lower_bounds(bounds[:,0])
    opt.set_upper_bounds(bounds[:,1])
    #if cons:
    #    opt.add_inequality_constraint(nlopt_constraint)
    opt.set_maxeval(100)
    res = opt.optimize(res)

    return res, opt.last_optimum_value()

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
    def __init__(self, kernel, bounds, noise=None, noiseGP=False, cons=None):
        self.kernel = kernel
        self.x = []
        self.y = []
        self.yd = []
        self.yn = []
        self.ydn = []
        self.bounds = np.array(bounds)
        self.ndim = self.bounds.shape[0]
        self.noise = noise
        self.cons = cons
        if noise is None:
            self.noise = [0., np.zeros(self.ndim)]
        self.noiseGP = None
        if noiseGP:
            kernel = gp_noder.SquaredExponentialKernel(kernel.L, 1.)
            self.noiseGP = [gp_noder.GaussianProcess(kernel, bounds, noise=0.1)]
            for i in range(0, self.ndim):
                kernel = gp_noder.SquaredExponentialKernel(kernel.L, 1.)
                self.noiseGP.append(gp_noder.GaussianProcess(kernel, bounds, noise=0.1))
    
    def evaluate(self, xs):
        xs = _sanitize(xs)
        x, y, yd = np.array(self.x), np.array(self.y), np.array(self.yd)
        d = np.concatenate((y, yd.flatten()))
        Ki = np.hstack((self.kernel.evaluate(xs, x), -self.kernel.gradient(xs, x)))

        K = self.kernel.evaluate(xs, xs)
        Kd = self.Kd
        Kd = Kd + 1e-30*np.diag(np.ones_like(np.diag(Kd)))
        L = sp.linalg.cho_factor(Kd)
        mu = np.dot(Ki, sp.linalg.cho_solve(L, d))
        cov = K - np.dot(Ki, sp.linalg.cho_solve(L, Ki.T))
        return mu, cov

    def evaluate_grad(self, xs):
        xs = _sanitize(xs)
        x, y, yd = np.array(self.x), np.array(self.y), np.array(self.yd)
        d = np.concatenate((y, yd.flatten()))
        #Ki = np.hstack((self.kernel.evaluate(xs, x), -self.kernel.gradient(xs, x)))
        Kid = self.kernel.gradient(xs, x)
        Ki = np.vstack((np.hstack((self.kernel.evaluate(xs, x), -self.kernel.gradient(xs, x))), np.hstack((-self.kernel.gradient(x, xs).T, self.kernel.hessian(xs, x)))))

        Kd = self.kernel.gradient(xs, xs)
        K = np.vstack((np.hstack((self.kernel.evaluate(xs, xs), -Kd)), np.hstack((-Kd.T, self.kernel.hessian(xs, xs)))))
        Kd = self.Kd + 1e-30*np.diag(np.ones_like(np.diag(self.Kd)))
        L = sp.linalg.cho_factor(Kd)
        mu = np.dot(Ki, sp.linalg.cho_solve(L, d))
        cov = K - np.dot(Ki, sp.linalg.cho_solve(L, Ki.T))
        return mu, cov

    def exponential(self, xs):
        mu, cov = self.evaluate(xs)
        emu = np.exp(mu + np.diag(cov)/2)
        ecov = np.outer(emu, emu)*(np.exp(cov)-1)
        return emu, ecov

    def explore(self, n, func, x=None):
        assert len(self.x) == 0

        bounds = self.bounds.T
        #x = lhs(self.ndim, samples=n, criterion='center')
        if x is None:
            if self.cons:
                xa = []
                while len(xa) < n:
                    x = lhs(self.ndim, samples=n, criterion='maximin')
                    x = bounds[[0]] + x*(bounds[[1]]-bounds[[0]])
                    xa.extend(filter(lambda y: self.cons(y) <= 0., x))
                x = xa[:n]
            else:
                x = lhs(self.ndim, samples=n, criterion='maximin')
                x = bounds[[0]] + x*(bounds[[1]]-bounds[[0]])
        res = func(x)
        if len(res) > 2:
            y, yd, yn, ydn = res
            if not isinstance(yn, float):
                yn = list(yn)
            if not isinstance(ydn, float):
                ydn = list(ydn)
            self.train(list(x), list(y), list(yd), yn, ydn)
        else:
            y, yd = res
            self.train(list(x), list(y), list(yd))

    def train(self, x, y, yd, yn=0., ydn=0.):
        if len(np.array(x).shape) == 1:
            x = [x]
            if len(np.array(yd).shape) == 1:
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
        if self.noiseGP is None:
            yn, ydn = self.noise
        if isinstance(yn, float):
            yn = list(yn*np.ones_like(np.array(y)))
        if isinstance(ydn, float):
            ydn = list(ydn*np.ones_like(np.array(yd)))
        yn = np.array(yn)
        ydn = np.array(ydn)
        if self.noiseGP is None:
            if len(self.x) != len(yn):
                yn = np.tile(yn[0], len(self.x))
                ydn = np.tile(ydn[0], (len(self.x), 1))
        else:
            self.noiseGP[0].train(x, np.log(yn/self.noise[0]))
            for i in range(1, 1 + self.ndim):
                ydn[ydn[:,i-1]==0.] = self.noise[1][i-1]
                self.noiseGP[i].train(x, np.log(ydn[:, i-1]/self.noise[1][i-1]))
            indices = np.indices((len(self.x), len(self.x)))
            yn = self.noiseGP[0].exponential(self.x)[0]*self.noise[0]
            ydn = []
            for i in range(1, 1 + self.ndim):
                ydn.append((self.noiseGP[i].exponential(self.x)[0]*self.noise[1][i-1]).reshape(-1,1))
        self.Kd += np.diag(np.concatenate((yn, np.hstack(ydn).flatten())))

    def posterior_min(self):
        res = _optimize(lambda x: self.evaluate(x)[0][0], self.bounds, cons=self.cons)
        return res + (self.evaluate(res[0])[1][0,0],)

    def get_noise(self, x):
        x = _sanitize(x)
        yn = self.noiseGP[0].exponential(x)[0]*self.noise[0]
        ydn = []
        for i in range(1, 1 + self.ndim):
            ydn.append((self.noiseGP[i].exponential(x)[0]*self.noise[1][i-1]).reshape(-1,1))
        return yn, ydn

    def data_min(self):
        ys = np.array(self.y).flatten()
        i = np.argmin(ys)
        return self.x[i], self.y[i]

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
        res = self.gp.posterior_min()
        self.fmin = res[1] + beta*self.gp.evaluate(res[0])[1][0][0]**0.5
        res = _optimize(lambda x: -self.evaluate(x)[0], self.gp.bounds, cons=self.gp.cons)
        return res[0]

sig = 0.2
def test_func(x):
    x = _sanitize(x)
    return np.sin(x).reshape((x.shape[0])) + sig*np.random.randn(),\
           np.cos(x).reshape((x.shape[0])) + sig*np.random.randn(),\
           sig**2*np.ones((x.shape[0], 1)), \
           sig**2*np.ones((x.shape[0], 1))


def _test_main():
    import matplotlib.pyplot as plt
    kernel = SquaredExponentialKernel([3.], 1.)
    bounds = [[0, 4*2*np.pi]]
    #kernel = SquaredExponentialKernel([1., 1.], 10.)
    #bounds = [[-10, 10], [-10.,10]]

    gp = GaussianProcess(kernel, bounds, noise=[sig**2, [sig**2]], noiseGP=True)
    gp.explore(4, test_func)
    xs = np.linspace(gp.bounds[0,0], gp.bounds[0,1], 500).reshape(-1,1)

    ei = ExpectedImprovement(gp)
    for i in range(0, 100):
        x = ei.optimize()
        y, yd, yn, ydn = test_func(x)
        gp.train(x, y, yd, yn, ydn)
        print gp.evaluate_grad(np.array([[1.], [3.], [7.]]))
        print gp.evaluate(np.array([[1.], [3.], [7.]]))

        #plt.ylim([-2,2])
        #plt.plot(xs, expected_improvement(xs))

        mu, cov = gp.evaluate(xs)
        std = np.diag(cov)**0.5
        #plt.contourf()
        plt.plot(xs.flatten(), mu)
        plt.fill_between(xs.flatten(), mu-std, mu + std, facecolor='gray')
        plt.scatter(gp.x, gp.y, c='k')

        plt.show()

if __name__ == "__main__":
    _test_main()
