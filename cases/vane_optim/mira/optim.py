#!/usr/bin/python2

import numpy as np
import os, sys, glob, shutil
import subprocess
import cPickle as pkl
from multiprocessing import Pool

import client
import gp as GP

homeDir = '/home/talnikar/adFVM/'
workDir = '/projects/LESOpt/talnikar/vane_optim/doe/'
#workDir = '/projects/LESOpt/talnikar/vane_optim/optim/'
appsDir = homeDir + 'apps/'
codeDir = workDir + 'gencode/'
initCaseFile = homeDir + 'templates/vane_optim.py'
primCaseFile = homeDir + 'templates/vane_optim_prim.py'
adjCaseFile = homeDir + 'templates/vane_optim_adj.py'
primal = os.path.join(appsDir, 'problem.py')
adjoint = os.path.join(appsDir, 'adjoint.py')
#eps = 1e-2
#eps = 1e-3
# from norm of new points displacement less than 1e-5 needed
eps = 1e-5

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

def spawnJob(args, output, error, cwd='.', block='BLOCK1'):
    nProcs = 8192
    nProcsPerNode = 16
    return subprocess.Popen(['runjob', 
                         '-n', str(nProcs), 
                         '-p', str(nProcsPerNode),
                         '--block', os.environ[block],
                         '--exp-env', 'BGLOCKLESSMPIO_F_TYPE', 
                         '--exp-env', 'PYTHONPATH',
                         '--exp-env', 'LD_LIBRARY_PATH',
                         '--verbose', 'INFO',
                         ':'] 
                        + args, cwd=cwd, stdout=output, stderr=error)

def readObjectiveFile(objectiveFile, gradEps):
    objective = []
    gradient = []
    gradientNoise = []
    index = 0
    with open(objectiveFile, 'r') as f:
        for line in f.readlines(): 
            words = line.split(' ')
            if words[0] == 'orig':
                objective = [float(words[-2]), float(words[-1])]
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
    index = currIndex
    if currIndex == -1: 
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
    paramCodeDir = paramDir + 'gencode'
    if not os.path.exists(paramCodeDir):
        shutil.copytree(codeDir, paramDir + 'gencode')
    shutil.copy(initCaseFile, paramDir)
    initFile = paramDir + os.path.basename(initCaseFile)
    shutil.copy(primCaseFile, paramDir)
    primalFile = paramDir + os.path.basename(primCaseFile)
    shutil.copy(adjCaseFile, paramDir)
    adjointFile = paramDir + os.path.basename(adjCaseFile)

    #shutil.copy(workDir + 'rerun/job.sh', paramDir)
    #shutil.copy(workDir + 'rerun/run.sh', paramDir)

    if runSimulation:
        if stateIndex <= 1:
            with open(paramDir + 'init_output.log', 'w') as f, open(paramDir + 'init_error.log', 'w') as fe:
                p1 = spawnJob([sys.executable, primal, initFile], f, fe, cwd=paramDir)
                ret = p1.wait()
                assert ret == 0
            update_state(state, 'PRIMAL1', currIndex)
        if stateIndex <= 2:
            with open(paramDir + 'primal_output.log', 'w') as f, open(paramDir + 'primal_error.log', 'w') as fe, \
                    open(paramDir + 'adjoint_output.log', 'w') as f2, open(paramDir + 'adjoint_error.log', 'w') as fe2:
                p1 = spawnJob([sys.executable, primal, primalFile], f, fe, cwd=paramDir)
                p2 = spawnJob([sys.executable, primal, adjointFile], f2, fe2, cwd=paramDir, block='BLOCK2')
                assert p2.wait() == 0
                for i in range(0, 4-1):
                    p2 = spawnJob([sys.executable, adjoint, adjointFile, '--matop'], f2, fe2, cwd=paramDir, block='BLOCK2')
                    assert p2.wait() == 0
                assert p1.wait() == 0
                exit(0)
            update_state(state, 'PRIMADJ', currIndex)
        ##if stateIndex <= 1:
        ##    spawnJob([sys.executable, adjoint, problemFile], cwd=paramDir)
        ##    update_state(state, 'PRIMAL1')
        #if stateIndex <= 2:
        #    spawnJob([sys.executable, primal, adjointFile], cwd=paramDir)
        #    update_state(state, 'PRIMAL2', currIndex)

        #if stateIndex <= 3:
        #    spawnJob([sys.executable, adjoint, adjointFile], cwd=paramDir)
        #    update_state(state, 'ADJOINT', currIndex)

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

def optim():
    
    orig_bounds = np.array([[0.,1.], [0,1], [0,1], [0,1]])
    L, sigma = 0.5, 0.001
    #L, sigma = 0.3, 0.003
    mean = 0.0092
    kernel = GP.SquaredExponentialKernel([L, L, L, L], sigma)
    gp = GP.GaussianProcess(kernel, orig_bounds, noise=[5e-9, [1e-9, 1e-7, 1e-8, 1e-7]], noiseGP=True, cons=constraint)
    ei = GP.ExpectedImprovement(gp)
    
    assert os.path.exists(stateFile)
    state = load_state()
    for index in range(0, len(state['state'])):
        if get_state(state, index) != 'DONE':
            x = state['points'][index]
            res = evaluate(x, state, index)
            state['evals'].append(res)
            update_state(state, 'DONE', index)
            exit(1)
    print state
    x = state['points']
    y = [res[0]-mean for res in state['evals']]
    yd = [res[2:6] for res in state['evals']]
    yn = [res[1] for res in state['evals']]
    ydn = [res[6:10] for res in state['evals']]
    #print yd
    #print ydn
    #print yd
    #print ydn
    #exit(1)

    gp.train(x, y, yd, yn, ydn)

    for i in range(len(state['points']), 100):
        dmin = gp.data_min()
        pmin = gp.posterior_min()
        print 'data min:', dmin[0], dmin[1] + mean
        print 'posterior min:', pmin[0], pmin[1] + mean

        x = ei.optimize()
        print 'ei choice:', i, x
        #exit(1)
        state['points'].append(x)
        state['state'].append('BEGIN')
        save_state(state)
        res = evaluate(x, state)
        print 'result:', res
        state['evals'].append(res)
        update_state(state, 'DONE')
        exit(1)
        resm = [x for x in res]
        resm[0] = res[0] - mean

        gp.train(x, *resm)

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
    #doe()


