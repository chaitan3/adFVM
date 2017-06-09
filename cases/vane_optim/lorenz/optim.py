#!/usr/bin/python2

import numpy as np
import os, sys, glob, shutil
import subprocess
import cPickle as pkl
from multiprocessing import Pool

import client
import gp as GP

#homeDir = '/home/talnikar/adFVM/'
#workDir = '/home/talnikar/adFVM/cases/vane_optim/test/'
homeDir = '/home/talnikar/adFVM-cpp/'
workDir = '/projects/LESOpt/talnikar/vane_optim/doe/'
appsDir = homeDir + 'apps/'
caseFile = homeDir + 'templates/vane_optim.py'
adjCaseFile = homeDir + 'templates/vane_optim_adj.py'
primal = os.path.join(appsDir, 'problem.py')
adjoint = os.path.join(appsDir, 'adjoint.py')
#eps = 1e-2
#eps = 1e-3
# from norm of new points displacement less than 1e-5 needed
eps = 1e-5

stateFile = 'state.pkl'
STATES = ['BEGIN', 'MESH', 'PRIMAL1', 'PRIMAL2', 'ADJOINT', 'DONE']
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
            if words[0] == 'orig' and len(objective) == 0:
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

def evaluate(param, state, currIndex=-1, genAdjoint=True, runSimulation=True):
#def evaluate(param, state, genAdjoint=True, runSimulation=False): 
    #return np.random.rand(),  np.random.rand(4), np.random.rand(), np.random.rand(4)
    stateIndex = STATES.index(get_state(state, currIndex))
    assert stateIndex <= 4
    index = len(state['points'])-1
    param = np.array(param)
    base = 'param{}/'.format(index)
    paramDir = os.path.join(workDir, base)

    # get mesh from remote server
    args = [(param, paramDir, base, True)]
    gradEps = []

    if not os.path.exists(paramDir):
        os.makedirs(paramDir)
    #get_mesh(args[0])
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
            if not os.path.exists(gradDir):
                os.makedirs(gradDir)
            args.append((perturbedParam, gradDir, base2, False))
            #get_mesh(args[-1])

    if stateIndex == 0:
        pool = Pool(len(args))
        res = pool.map(get_mesh, args)
        update_state(state, 'MESH', currIndex)

    # copy caseFile
    shutil.copy(caseFile, paramDir)
    problemFile = paramDir + os.path.basename(caseFile)
    shutil.copy(adjCaseFile, paramDir)
    adjointFile = paramDir + os.path.basename(adjCaseFile)

    if runSimulation:
        if stateIndex <= 1:
            spawnJob([sys.executable, primal, problemFile], cwd=paramDir)
            update_state(state, 'PRIMAL1', currIndex)
        #if stateIndex <= 1:
        #    spawnJob([sys.executable, adjoint, problemFile], cwd=paramDir)
        #    update_state(state, 'PRIMAL1')

        if stateIndex <= 2:
            spawnJob([sys.executable, primal, adjointFile], cwd=paramDir)
            update_state(state, 'PRIMAL2', currIndex)

        if stateIndex <= 3:
            spawnJob([sys.executable, adjoint, adjointFile], cwd=paramDir)
            update_state(state, 'ADJOINT', currIndex)

        return readObjectiveFile(os.path.join(paramDir, 'objective.txt'), gradEps)
    return

def constraint(x):
    return sum(x) - 1.

def doe():
    xe = np.array([[.99,0,0,0],
                   [0,.99,0,0],
                   [0,0,.99,0],
                   [0,0,0,.99],
                   [0,0,0,0],
                   [0.24,0.24,0.24,0.24],

                   #[0.49, 0.49, 0, 0]
                   #[0.33, 0.33, 0.33, 0]
        ])
    state = {'points':[], 'evals':[], 'state': []}
    for i in range(0, len(xe)):
        x = xe[i]
        state['points'].append(x)
        state['state'].append('BEGIN')
        save_state(state)
        evaluate(x, state, runSimulation=False)
        #res = evaluate(x, state)
        #state['evals'].append(res)
        #update_state(state, 'DONE')

def optim():
    
    orig_bounds = np.array([[0.,1.], [0,1], [0,1], [0,1]])
    L = 0.25
    sigma = 1.
    kernel = GP.SquaredExponentialKernel([L, L, L, L], sigma)
    gp = GP.GaussianProcess(kernel, orig_bounds, noise=[0.1, [0.5, 0.5, 0.5, 0.5]], noiseGP=True, cons=constraint)
    ei = GP.ExpectedImprovement(gp)
    
    assert os.path.exists(stateFile)
    state = load_state()
    for index in range(0, len(state['state'])):
        if get_state(state, index) != 'DONE':
            x = state['points'][index]
            res = evaluate(x, state, index)
            state['evals'].append(res)
            update_state(state, 'DONE', index)
    x = state['points']
    y = [res[0] for res in state['evals']]
    yd = [res[1] for res in state['evals']]
    yn = [res[2] for res in state['evals']]
    ydn = [res[3] for res in state['evals']]
    gp.train(x, y, yd, yn, ydn)

    for i in range(len(state['points']), 100):
        print 'data min:', gp.data_min()
        print 'posterior min:', gp.posterior_min()

        x = ei.optimize()
        print 'ei choice:', i, x
        state['points'].append(x)
        state['state'].append('BEGIN')
        save_state(state)
        res = evaluate(x, state)
        print 'result:', res
        state['evals'].append(res)
        update_state(state, 'DONE')

        gp.train(x, *res)

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
    #optim()
    doe()


