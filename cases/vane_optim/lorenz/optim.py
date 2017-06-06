#!/usr/bin/python2

import numpy as np
import os, sys, glob, shutil
import subprocess
import cPickle as pkl
from multiprocessing import Pool

from adFVM import config
import client
import gp as GP

appsDir = '/home/talnikar/adFVM/apps/'
workDir = '/home/talnikar/adFVM/cases/vane_optim/test/'
caseFile = '/home/talnikar/adFVM/templates/vane_optim.py'

primal = os.path.join(appsDir, 'problem.py')
adjoint = os.path.join(appsDir, 'adjoint.py')
paramHistory = []
eps = 0.01

def spawnJob(args, cwd='.'):
    #nProcs = 4096
    #nProcsPerNode = 16
    nProcs = 16
    with open(cwd + 'output.log', 'w') as f, open(cwd + 'error.log', 'w') as fe:
        subprocess.check_call(['mpirun', '-np', str(nProcs)] + args, cwd=cwd, stdout=f, stderr=fe)
        #subprocess.check_call(['runjob', 
        #                 '-n', str(nProcs), 
        #                 '-p', str(nProcsPerNode),
        #                 '--block', os.environ['COBALT_PARTNAME'],
        #                 #'--exp-env', 'BGLOCKLESSMPIO_F_TYPE', 
        #                 #'--exp-env', 'PYTHONPATH',
        #                 '--env_all',
        #                 '--verbose', 'INFO',
        #                 ':'] 
        #                + args, cwd=cwd stdout=f, stderr=fe)

def readObjectiveFile(objectiveFile, gradEps):
    objective = []
    gradient = []
    gradientNoise = []
    index = 0
    with open(objectiveFile, 'r') as f:
        for line in f.readlines(): 
            words = line.split(' ')
            if words[0] == 'orig':
                objective += [float(words[-2]), float(words[-1])]
            elif words[0] == 'adjoint':
                per = gradEps[index]
                gradient.append(float(words[-2])/per)
                gradientNoise.append(float(words[-1])/(per**2))
                index += 1
    assert len(objective) > 0
    assert len(gradient) > 0
    return objective + gradient + gradientNoise

def get_mesh(args):
    client.get_mesh(*args)

#def evaluate(param, genAdjoint=True, runSimulation=True):
def evaluate(param, genAdjoint=False, runSimulation=False):
    param = np.array(param)
    index = len(paramHistory)
    paramHistory.append(param)
    base = 'param{}/'.format(index)
    paramDir = os.path.join(workDir, base)

    # get mesh from remote server
    os.makedirs(paramDir)
    args = [(param, paramDir, base, True)]
    #get_mesh(args[0])

    gradEps = []

    if genAdjoint:
        for index in range(0, len(param)):
            perturbedParam = param.copy()
            if sum(param) + eps > 1.:
                if param[index]-eps >= 0.:
                    perturbedParam[index] -= eps
                gradEps.append(-eps)
            else:
                perturbedParam[index] += eps
                gradEps.append(eps)

            base2 = base + 'grad{}/'.format(index)
            gradDir = os.path.join(workDir, base2)
            os.makedirs(gradDir)
            args.append((perturbedParam, gradDir, base2, False))
            #get_mesh(args[-1])

    pool = Pool(len(args))
    res = pool.map(get_mesh, args)

    # copy caseFile
    shutil.copy(caseFile, paramDir)
    problemFile = paramDir + os.path.basename(caseFile)

    if runSimulation:
        spawnJob([sys.executable, primal, problemFile], cwd=paramDir)
        spawnJob([sys.executable, adjoint, problemFile], cwd=paramDir)
        return readObjectiveFile(os.path.join(paramDir, 'objective.txt'), gradEps)
    return

#from adFVM.optim import designOfExperiment
#print designOfExperiment(lambda x: evaluate(x, False, False), paramBounds, 2*nParam)
#print evaluate(np.zeros(8)*1., False, False)


def constraint(x):
    return sum(x) - 1.

def optim():
    #print evaluate([0.15,0.15,0.3,0.14])

        # exploration? deterministic?
    #gp.explore(16, evaluate)
    xe = np.array([[.99,0,0,0],
                   [0,.99,0,0],
                   [0,0,.99,0],
                   [0,0,0,.99],
                   [0,0,0,0],
                   [0.24,0.24,0.24,0.24],
                   #[0.49, 0.49, 0, 0]
                   #[0.33, 0.33, 0.33, 0]
        ])

    for i in range(0, 10):
        evaluate(xe[i])
    #gp.explore(10, evaluate, x=xe)

    values = []
    evals = []
    gps = []

    orig_bounds = np.array([[0.,1.], [0,1], [0,1], [0,1]])
    L = 0.25
    sigma = 1.
    kernel = GP.SquaredExponentialKernel([L, L, L, L], sigma)
    gp = GP.GaussianProcess(kernel, orig_bounds, noise=[0.1, [0.5, 0.5, 0.5, 0.5]], noiseGP=True, cons=constraint)
    ei = GP.ExpectedImprovement(gp)
    

    for i in range(0, 100):
        res = gp.posterior_min()
        gps.append(res)
        print 'data min:', gp.data_min()
        print 'posterior min:', res
        print 'ei choice:', i, x

        x = ei.optimize()
        y, yd, yn, ydn = evaluate(x)
        gp.train(x, y, yd, yn, ydn)

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
        print

    #with open('{}.pkl'.format(optime), 'w') as f:
    #    pkl.dump([evals, gps, values], f)

if __name__ == "__main__":
    optim()
    #minimum()


