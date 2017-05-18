import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

from ar import arsel
#from clorenz import lorenz
import lorenz
import gp as GP

dimension = 2

orig_bounds = np.array([[30.,70.], [0.5, 3.5]])
# if threshold is negative constraint passes all points
constraint = lambda points: np.linalg.norm(points - np.array([-1,-1]), axis=1) > -0.2

#default_params = np.array([50., 2.])
default_params = np.array([60., 1.])
param_names = ['rho', 'beta']

# value = 0.03019
#optimum = np.array([-0.049364162999999905, -0.08275111066666652])
optimum = 0.03019

number_of_jobs = 30
parallel = 4

min_T = 300
burnin = 2.

def objective(args):
    pool = Pool(len(args))
    data = pool.map(objective_single, args)
    return [x[0] for x in data], [x[1] for x in data]

def objective_single(params):
    p1 = params[0]
    p2 = 10.
    p3 = params[1]

    print 'lorenz: rho = {0}, sigma = {1}, beta = {2}'.format(p1, p2, p3)
    dt = 0.01
    T = min_T
    data = lorenz.lss_lorenz(p1, p2, p3, dt, T)
    print data
    return data
    #mean, scale = 5e3, 5e3
    #scaled = (obj - mean)/scale

    #mu = obj.mean()
    #var = arsel(obj).mu_sigma[0]**2
    #return mu, var, T/min_T

#def main(jobid, params):
#    new_params = []
#    for param in param_names:
#        new_params.append(params[param][0])
#    #return objective_single([new_params, jobid])[:2]
#    return objective_single([new_params, jobid])[0]

def optim():
    kernel = GP.SquaredExponentialKernel([10., 1.], 1.)
    gp = GP.GaussianProcess(kernel, orig_bounds)
    gp.explore(4, objective)
    x1 = np.linspace(gp.bounds[0,0], gp.bounds[0,1], 40)
    x2 = np.linspace(gp.bounds[1,0], gp.bounds[1,1], 40)
    x1, x2 = np.meshgrid(x1, x2)
    xs = np.vstack((x1.flatten(), x2.flatten())).T

    # test noise and multid support

    ei = GP.ExpectedImprovement(gp)
    for i in range(0, 100):
        x = ei.optimize()
        y, yd = objective_single(x)
        gp.train(x, y, yd, yn=1e-2, ydn=1e-2)

        #plt.ylim([-2,2])
        #plt.scatter(gp.x, gp.y, c='k')
        ##plt.plot(xs, expected_improvement(xs))

        mu, cov = gp.evaluate(xs)
        plt.contourf(x1, x2, mu.reshape(x1.shape), 100)
        plt.colorbar()
        #std = np.diag(cov)**0.5
        #plt.plot(xs.flatten(), mu)
        #plt.fill_between(xs.flatten(), mu-std, mu + std, facecolor='gray')

        plt.show()

if __name__ == "__main__":
    optim()
