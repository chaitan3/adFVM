import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import cPickle as pkl
from functools import partial
from scipy.optimize import rosen, rosen_der

#from clorenz import lorenz
import lorenz
import gp as GP
import sys

GP.beta = float(sys.argv[1])
#optime = 'optim_nd2_lorenz'
#optime = 'optim_nd1_lorenz'
#optime = 'beta_{}/optim_nd2_rosenbrock7'.format(GP.beta)
optime = 'beta_{}/optim_nd2_rastigrin3'.format(GP.beta)
#import gp_noder as GP
#optim = 'optim_noder'

# if threshold is negative constraint passes all points

sig = 0.25
#sig = 1.
#sig = 4.

# rosenbrock
#def objective_single(x):
#    x = GP._sanitize(x)
#    y = []
#    yd = []
#    for i in range(0, x.shape[0]):
#        y.append(rosen(x[i]))
#        yd.append(rosen_der(x[i]))
#    return np.array(y) + sig*np.random.randn(x.shape[0]),\
#           np.array(yd) + sig*np.random.randn(*x.shape),\
#           sig**2, \
#           sig**2

# rastigrin
#orig_bounds = np.array([[-3., 3], [-3, 3.]])
orig_bounds = np.array([[-10., 10.], [-10., 10.]])
def objective_single(x):
    x = GP._sanitize(x)
    y = []
    yd = []
    fac = 1.5
    for i in range(0, x.shape[0]):
        f = 20 - 50 + x[i,0]**2 - 10*np.cos(fac*np.pi*x[i,0]) \
                    + x[i,1]**2 - 10*np.cos(fac*np.pi*x[i,1])
        fd = np.array([
              2*x[i,0] + 10*fac*np.pi*np.sin(fac*np.pi*x[i,0]),
              2*x[i,1] + 10*fac*np.pi*np.sin(fac*np.pi*x[i,1])
            ])
        y.append(f)
        yd.append(fd)
    return np.array(y) + sig*np.random.randn(x.shape[0]),\
           np.array(yd) + sig*np.random.randn(*x.shape),\
           sig**2, \
           sig**2


#orig_bounds = np.array([[-5.12, 5.12], [-5.12, 5.12]])
def minimum():
    #print objective_single([58.,1.24])
    #print objective_single([51.8,1.8])
    x1 = np.linspace(orig_bounds[0,0], orig_bounds[0,1], 40)
    #x1 = [orig_bounds[0,0] + 1]
    x2 = np.linspace(orig_bounds[1,0], orig_bounds[1,1], 40)
    #x2 = [2.5 + 0.05]
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
    #plt.errorbar(x1, ys, yerr=yns)
    #for x, y, yd in zip(x1, ys, yds):
    #    xd = np.linspace(x-0.1, x+0.1, 100)
    #    yd = y + yd[0]*(xd-x)
    #    plt.plot(xd, yd)
    #plt.show()

def optim():
    values = []
    evals = []
    gps = []
    #nj = 1500
    #ne = 35
    nj = 1000
    ne = 50
    for j in range(0, nj):
        evals.append([])
        gps.append([])
        print j
        fail = False

        #kernel = GP.SquaredExponentialKernel([2.], 1.)#, [1e-2, 1e-2])
        #gp = GP.GaussianProcess(kernel, orig_bounds, noise=[1e-4, [1e-5]], noiseGP=True)
        #gp.explore(2, objective)

        #kernel = GP.SquaredExponentialKernel([2., 0.1], 1.)#, [1e-2, 1e-2])
        #def constraint(x):
        #    return sum(x) - 30.6
        #gp = GP.GaussianProcess(kernel, orig_bounds, noise=[1e-4, [1e-5, 1e-3]], noiseGP=True, cons=constraint)
        #gp.explore(4, objective)
        #kernel = GP.SquaredExponentialKernel([1., 1.], 100)#, [1e-2, 1e-2])
        kernel = GP.SquaredExponentialKernel([1., 1.], 50.)#, [1e-2, 1e-2])
        gp = GP.GaussianProcess(kernel, orig_bounds, noise=[sig**2, [sig**2, sig**2]], noiseGP=True)
        #gp = GP.GaussianProcess(kernel, orig_bounds, noise=[sig**2, sig**2], noiseGP=False)
        gp.explore(4, objective_single)
        #gp.explore(100, objective_single)

        x1 = np.linspace(gp.bounds[0,0], gp.bounds[0,1], 40)
        xs = x1.reshape(-1,1)
        x2 = np.linspace(gp.bounds[1,0], gp.bounds[1,1], 40)
        x1, x2 = np.meshgrid(x1, x2)
        xs = np.vstack((x1.flatten(), x2.flatten())).T

        ei = GP.ExpectedImprovement(gp)

        # acquisition function improvements:
        # * noise: using EI with std for now
        # * batch
        # * finite budget
        #print

        for i in range(0, ne):
            print j, i
            x = ei.optimize()
            #try:
            #    x = ei.optimize()
            #except:
            #    fail = True
            #    break
            #x = ei.optimize()
            evals[-1].append(gp.data_min())
            gps[-1].append(gp.posterior_min())
            #print 'ei choice:', i, x
            #print 'data min', evals[-1][-1]
            #print 'posterior min', gps[-1][-1]

            #eix = ei.evaluate(xs)
            #plt.contourf(x1, x2, eix.reshape(x1.shape), 100)
            #plt.colorbar()
            #plt.savefig('debug/ei_{}.pdf'.format(i))
            #plt.clf()
            #plt.show()

            #mu, cov = gp.evaluate(xs)
            #plt.contourf(x1, x2, mu.reshape(x1.shape), 100)
            #plt.colorbar()
            #plt.savefig('debug/mu_{}.pdf'.format(i))
            #plt.clf()
            #plt.show()

            y, yd, yn, ydn = objective_single(x)
            #y, yd, yn, ydn = test_func(x)
            gp.train(x, y, yd, yn, ydn)
            #print
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
