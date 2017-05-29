import numpy as np
#import matplotlib.pyplot as plt
from multiprocessing import Pool
import cPickle as pkl
from functools import partial
from scipy.optimize import rosen, rosen_der

#from clorenz import lorenz
import lorenz
import gp as GP
#optime = 'optim_nd2_lorenz'
optime = 'optim_nd1_lorenz'
#optime = 'optim_nd2_rosenbrock'
#import gp_noder as GP
#optim = 'optim_noder'

#orig_bounds = lorenz.orig_bounds
orig_bounds = lorenz.orig_bounds[:1]
# if threshold is negative constraint passes all points
param_names = ['rho', 'beta']

min_T = 50
#min_T = 300
burnin = 2.
#reps = 10
reps = 8
parallel = True

def objective(args):
    data = []
    for arg in args:
        data.append(objective_single(arg))
    y = [x[0] for x in data]
    yd = [x[1] for x in data]
    yn = [x[2] for x in data]
    ydn = [x[3] for x in data]
    return y, yd, yn, ydn
pool = Pool(reps)
def objective_single(params):
    p1 = params[0]
    p2 = 10.
    #p3 = params[1]
    p3 = 2.55

    print 'lorenz: rho = {0}, sigma = {1}, beta = {2}'.format(p1, p2, p3)
    dt = 0.01
    T = min_T
    if parallel:
        lss_lorenz = partial(lorenz.lss_lorenz, p1, p2, p3, dt, T)
        rng = [np.random.rand(3) for i in range(0, reps)]
        data = pool.map(lss_lorenz, rng)
        ys = [x[0] for x in data]
        #yds = [x[1] for x in data]
        yds = [x[1][:1] for x in data]
    else:
        ys = []
        yds = []
        for i in range(0, reps):
            y, yd = lorenz.lss_lorenz(p1, p2, p3, dt, T)
            ys.append(y)
            yds.append(yd)
    ys = np.array(ys)
    yds = np.array(yds)
    y = np.mean(ys)
    yd = np.mean(yds, axis=0)
    yn = np.var(ys)
    ydn = np.var(yds, axis=0)
    print 'value:', y, yd, yn, ydn
    return y, yd, yn, ydn
    #mean, scale = 5e3, 5e3
    #scaled = (obj - mean)/scale

    #mu = obj.mean()
    #var = arsel(obj).mu_sigma[0]**2
    #return mu, var, T/min_T

sig = 10.
def test_func(x):
    x = GP._sanitize(x)
    y = []
    yd = []
    for i in range(0, x.shape[0]):
        y.append(rosen(x[i]))
        yd.append(rosen_der(x[i]))
    return np.array(y) + sig*np.random.randn(x.shape[0]),\
           np.array(yd) + sig*np.random.randn(*x.shape),\
           sig**2, \
           sig**2

def minimum():
    #print objective_single([58.,1.24])
    #print objective_single([51.8,1.8])
    x1 = np.linspace(orig_bounds[0,0], orig_bounds[0,1], 40)
    #x1 = [orig_bounds[0,0] + 1]
    x2 = np.linspace(orig_bounds[1,0], orig_bounds[1,1], 40)
    x2 = [orig_bounds[1, 0] + 0.05]
    miny = np.inf
    minx = None
    ys = []
    yns = []
    yds = []
    for y1 in x1:
        for y2 in x2:
            x = [y1, y2]
            y, yd, yn, _ = objective_single(x)
            ys.append(y)
            yds.append(yd)
            yns.append(yn**0.5)
            if y < miny:
                miny = y
                minx = x
                print x, y
    #plt.plot(x1, ys)
    plt.errorbar(x1, ys, yerr=yns)
    for x, y, yd in zip(x1, ys, yds):
        xd = np.linspace(x-0.1, x+0.1, 100)
        yd = y + yd[0]*(xd-x)
        plt.plot(xd, yd)
    plt.show()

def optim():
    kernel = GP.SquaredExponentialKernel([2.], 1.)#, [1e-2, 1e-2])
    gp = GP.GaussianProcess(kernel, orig_bounds, noise=[1e-4, [1e-5]], noiseGP=True)
    gp.explore(2, objective)

    #kernel = GP.SquaredExponentialKernel([2., 0.1], 1.)#, [1e-2, 1e-2])
    #gp = GP.GaussianProcess(kernel, orig_bounds, noise=[1e-4, [1e-5, 1e-3]], noiseGP=True)
    #gp.explore(4, objective)
    #orig_bounds = np.array([[-3., 3], [-3, 3.]])
    #kernel = GP.SquaredExponentialKernel([1., 1], 100.)#, [1e-2, 1e-2])
    #gp = GP.GaussianProcess(kernel, orig_bounds, noise=[sig**2, sig**2])
    #gp.explore(4, test_func)

    x1 = np.linspace(gp.bounds[0,0], gp.bounds[0,1], 40)
    #xs = x1.reshape(-1,1)
    x2 = np.linspace(gp.bounds[1,0], gp.bounds[1,1], 40)
    x1, x2 = np.meshgrid(x1, x2)
    xs = np.vstack((x1.flatten(), x2.flatten())).T

    ei = GP.ExpectedImprovement(gp)

    # acquisition function improvements:
    # * noise: using EI with std for now
    # * batch
    # * finite budget

    print
    values = []
    evals = []
    gps = []
    nj = 100
    for j in range(0, nj):
        evals.append([])
        gps.append([])
        print optime, j
        fail = False
        for i in range(0, 100):
            x = ei.optimize()
            try:
                x = ei.optimize()
            except:
                fail = True
                break
            evals[-1].append(gp.data_min())
            gps[-1].append(gp.posterior_min())
            print 'ei choice:', i, x

            #eix = ei.evaluate(xs)
            #plt.plot(x1, eix.reshape(x1.shape))
            #plt.contourf(x1, x2, eix.reshape(x1.shape), 100)
            #plt.colorbar()
            ##plt.savefig('ei_{}.pdf'.format(i))
            ##plt.clf()
            #plt.show()

            #mu, cov = gp.evaluate(xs)
            ##mu, cov = gp.noiseGP[1].exponential(xs)
            ##plt.plot(x1, mu*gp.noise[0])
            ##plt.show()
            ##std = np.diag(cov)**0.5
            #plt.contourf(x1, x2, mu.reshape(x1.shape), 100)
            #plt.colorbar()
            ###plt.savefig('mu_{}.pdf'.format(i))
            ###plt.clf()
            #plt.show()
            #plt.contourf(x1, x2, std.reshape(x1.shape), 1000)
            #plt.colorbar()
            #plt.savefig('cov_{}.pdf'.format(i))
            #plt.clf()

            y, yd, yn, ydn = objective_single(x)
            #y, yd, yn, ydn = test_func(x)
            gp.train(x, y, yd, yn, ydn)
            print
        if not fail:
            values.append(gp.y)
        else:
            evals = evals[:-1]
            gps = gps[:-1]
        with open('{}.pkl'.format(optime), 'w') as f:
            pkl.dump([evals, gps, values], f)

if __name__ == "__main__":
    optim()
    #minimum()
