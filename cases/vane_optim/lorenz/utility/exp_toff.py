#!/usr/bin/python2

import numpy as np
import os, sys, glob, shutil
import subprocess
import cPickle as pkl
from multiprocessing import Pool
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('../')
#import gp as GP
import gp_noder as GP
import gp as GP2

stateFile = 'state.pkl'
STATES = ['BEGIN', 'MESH', 'PRIMAL1', 'PRIMADJ', 'DONE']
def save_state(state):
    with open(stateFile, 'w') as f:
        pkl.dump(state, f)
def load_state():
    with open(stateFile, 'r') as f:
        return pkl.load(f)
def update_state(state, value, index=-1):
    state['state'][index] = value
    save_state(state)
def get_state(state, index=-1):
    return state['state'][index]

def constraint(x):
    return sum(x) - 1.

def doe():
    xe = np.array([[.99,0,0,0],
                   [0,.99,0,0],
                   [0,0,.99,0],
                   [0,0,0,.99],
                   [0,0,0,0],
                   [0.24,0.24,0.24,0.24],
                   [0.49, 0., 0.49, 0],
                   [0., 0.49, 0.49, 0],
                   #[0.33, 0.33, 0.33, 0]
        ])
    if os.path.exists(stateFile):
        state = load_state()
        for index in range(0, len(state['state'])):
            if get_state(state, index) != 'DONE':
                x = state['points'][index]
                res = evaluate(x, state, index)
                state['evals'].append(res)
                update_state(state, 'DONE', index)
                exit(0)

    else:
        state = {'points':[], 'evals':[], 'state': []}
    for i in range(len(state['points']), len(xe)):
        x = xe[i]
        state['points'].append(x)
        state['state'].append('BEGIN')
        save_state(state)
        #res = evaluate(x, state, runSimulation=False)
        res = evaluate(x, state)
        state['evals'].append(res)
        update_state(state, 'DONE')
        exit(0)

import sys
grad = len(sys.argv) > 1
#grad = True

def optim():
    
    orig_bounds = np.array([[0.,1.], [0,1], [0,1], [0,1]])
    L, sigma = 0.5, 0.001
    #L, sigma = 0.3, 0.003
    mean = 0.0092
    if grad:
        kernel = GP2.SquaredExponentialKernel([L, L, L, L], sigma)
        gp = GP2.GaussianProcess(kernel, orig_bounds, noise=[5e-9, [1e-9, 1e-7, 1e-8, 1e-7]], noiseGP=True, cons=constraint)
        ei = GP2.ExpectedImprovement(gp)
    else:
        kernel = GP.SquaredExponentialKernel([L, L, L, L], sigma)
        gp = GP.GaussianProcess(kernel, orig_bounds, noise=5e-9, noiseGP=True, cons=constraint)
        ei = GP.ExpectedImprovement(gp)

    kernel = GP2.SquaredExponentialKernel([L, L, L, L], sigma)
    gp2 = GP2.GaussianProcess(kernel, orig_bounds, noise=[5e-9, [1e-9, 1e-7, 1e-8, 1e-7]], noiseGP=True, cons=constraint)
    #ei = GP2.ExpectedImprovement(gp)
    
    assert os.path.exists(stateFile)
    state = load_state()
    #for index in range(0, len(state['state'])):
    #    if get_state(state, index) != 'DONE':
    #        x = state['points'][index]
    #        res = evaluate(x, state, index)
    #        state['evals'].append(res)
    #        update_state(state, 'DONE', index)
    #        exit(1)
    n = len(state['points'])
    if len(state['evals']) < len(state['points']):
        n -= 1
    m = 8
    x = state['points'][:n]
    y = [res[0]-mean for res in state['evals']][:n]
    yd = [res[2:6] for res in state['evals']][:n]
    yn = [res[1] for res in state['evals']][:n]
    ydn = [res[6:10] for res in state['evals']][:n]
    #print yd
    #print ydn
    #print yd
    #print ydn
    #exit(1)
    gp2.train(x[:m], y[:m], yd[:m], yn[:m], ydn[:m])
    mus = []
    varis = []
    for i in range(m, n):
        mu, vari = gp2.evaluate(x[i])
        mus.append(mu[0] + mean)
        varis.append(vari[0,0]**0.5)
        gp2.train(x[i], y[i], yd[i], yn[i], ydn[i])
        # GRAD based optim
        #pmin = gp2.posterior_min()
        #print pmin[1] + mean, pmin[2]
    mus = np.array(mus)
    varis = np.array(varis)
    cm = plt.cm.get_cmap('RdYlBu')
    z = np.arange(0,len(mus))
    plt.locator_params(axis='x', numticks=4)
    plt.locator_params(axis='y', numticks=4)
    sc = plt.scatter(varis, mus, c=z, s=100)
    for i, x in enumerate(z):
        plt.annotate(x, (varis[i] + 1e-5, mus[i]), fontsize=16)
    plt.colorbar(sc, ticks=z+1)
    d = 0.0001
    plt.xlim([varis.min()-d, varis.max()+d])
    plt.ylim([mus.min()-d, mus.max()+d])
    plt.xlabel('standard deviation of GP at evaluation')
    plt.ylabel('mean of GP at evaluation')
    plt.show()
    exit(1)
    # GRAD based optim

    x = x[:m]
    y = y[:m]
    yn = yn[:m]
    yd = yd[:m]
    ydn = ydn[:m]

    if grad:
        gp.train(x, y, yd, yn, ydn)
    else:
        gp.train(x, y, yn)

    for i in range(m, 25):
        dmin = gp.data_min()
        pmin = gp.posterior_min() 
        #print 'data min:', dmin[0], dmin[1] + mean
        #print 'posterior min:', pmin[0], pmin[1] + mean
        print  '{},{},{},{}'.format(i, ','.join(np.char.mod('%f', pmin[0])), pmin[1] + mean, pmin[2])

        x = ei.optimize()

        if grad:
            y, _ = gp2.evaluate_grad(x)
            y, yd = y[0], y[1:]
            yn, ydn = gp2.get_noise(x)
            yn = yn[0]
            ydn = np.array(ydn).flatten()
            z = np.random.randn()*np.sqrt(yn) + y
            zd = np.random.randn(ydn.shape[0])*np.sqrt(ydn) + yd
            gp.train(x, z, zd, yn, ydn)
        else:
            y, _ = gp2.evaluate(x)
            y = y[0]
            yn = gp2.get_noise(x)[0]
            z = np.random.randn()*np.sqrt(yn) + y
            gp.train(x, z, yn)

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

    #with open('{}.pkl'.format(optime), 'w') as f:
    #    pkl.dump([evals, gps, values], f)

if __name__ == "__main__":
    optim()
    #doe()


