#!/usr/bin/python2

import numpy as np
import os, sys, glob, shutil
import subprocess

from adFVM import config
import client

appsDir = '/home/talnikar/adFVM/apps/'
workDir = '/home/talnikar/adFVM/cases/vane_optim/test/'
caseFile = '/home/talnikar/adFVM/templates/vane_optim.py'

primal = os.path.join(appsDir, 'problem.py')
adjoint = os.path.join(appsDir, 'adjoint.py')
paramHistory = []
eps = 1e-5

def spawnJob(args, cwd='.'):
    import subprocess
    return subprocess.check_call(args, cwd=cwd)
    #subprocess.call(args)
    nProcs = 4096
    #nProcs = 16
    nProcsPerNode = 16
    #subprocess.check_call(['mpirun', '-np', nProcs] + args, cwd=cwd)
    with open('output.log', 'w') as f, open('error.log', 'w') as fe:
        subprocess.check_call(['runjob', 
                         '-n', str(nProcs), 
                         '-p', str(nProcsPerNode),
                         '--block', os.environ['COBALT_PARTNAME'],
                         #'--exp-env', 'BGLOCKLESSMPIO_F_TYPE', 
                         #'--exp-env', 'PYTHONPATH',
                         '--env_all',
                         '--verbose', 'INFO',
                         ':'] 
                        + args, stdout=f, stderr=fe)

def readObjectiveFile(objectiveFile):
    objective = None
    gradient = []
    gradientNoise = []
    with open(objectiveFile, 'r') as f:
        for line in f.readlines(): 
            words = line.split(' ')
            if words[0] == 'objective':
                objective = [float(words[-2]), float(words[-1])]
                
            elif words[0] == 'adjoint':
                gradient.append(float(words[-2])/eps)
                gradientNoise.append(float(words[-1])/(eps**2))
    assert objective
    assert len(gradient) > 0
    return objective + gradient + gradientNoise

def evaluate(param, genAdjoint=True, runSimulation=True):
    index = len(paramHistory)
    paramHistory.append(param)
    paramDir = os.path.join(workDir, 'param{}/'.format(index))

    # get mesh from remote server
    os.makedirs(paramDir)
    client.get_mesh(param, paramDir)
    
    # copy caseFile
    shutil.copy(caseFile, paramDir)
    problemFile = paramDir + os.path.basename(caseFile)

    if genAdjoint:
        for index in range(0, len(param)):
            perturbedParam = param.copy()
            perturbedParam[index] += eps
            gradDir = os.path.join(paramDir, 'grad{}'.format(index))
            os.makedirs(gradDir)
            try:
                genMeshParam(perturbedParam, gradDir)
            except (OSError, subprocess.CalledProcessError) as e:
                print('Gen adjoint mesh param failed')
                raise

    if runSimulation:
        spawnJob([sys.executable, primal, problemFile])
        spawnJob([sys.executable, adjoint, problemFile])

    return readObjectiveFile(os.path.join(paramDir, 'objective.txt'))

#from adFVM.optim import designOfExperiment
#print designOfExperiment(lambda x: evaluate(x, False, False), paramBounds, 2*nParam)
#print evaluate(np.zeros(8)*1., False, False)


def constraint(x):
    return sum(x) - 1.

def optim():
    orig_bounds = np.array([[0.,1.], [0,1], [0,1], [0,1]])
    L = 0.25
    sigma = 1.
    kernel = GP.SquaredExponentialKernel([L, L, L, L], sigma)
    gp = GP.GaussianProcess(kernel, orig_bounds, noise=[0.1, [0.5, 0.5, 0.5, 0.5]], noiseGP=True, cons=constraint)
    ei = GP.ExpectedImprovement(gp)
    
    # exploration? deterministic?
    #gp.explore(16, evaluate)

    values = []
    evals = []
    gps = []

    for i in range(0, 100):
        res = gp.posterior_min()
        gps.append(res)
        print 'data min:', gp.data_min())
        print 'posterior min:', res)
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


