#!/usr/bin/python2
import numpy as np
import os
import shutil
import pickle
from pathos.multiprocessing import Pool
from pathos.helpers import mp

from adFVM.interface import SerialRunner
#class SerialRunner(object):
class SerialRunnerLorenz(object):
    def __init__(self, base, time, dt, templates, **kwargs):
        self.base = base
        self.time = time
        self.dt = dt
        self.internalCells = np.ones(3)

    def readFields(self, base, time):
        fields = np.random.rand(3)
        return self.runPrimal(fields, (0.0, int(self.time/self.dt)), '')[0]

    def copyCase(self, case):
        pass

    def removeCase(self, case):
        pass

    def runPrimal(self, fields, (parameter, nSteps), case):
        print case
        rho, beta, sigma = 28. + parameter, 2./3, 10.
        for i in range(0, nSteps):
            x, y, z = fields
            fields += self.dt*np.array([
                    sigma*(y-x),
                    x*(rho-z)-y,
                    x*y - beta*z
                ])
        return fields, 0.

    def runAdjoint(self, fields, (parameter, nSteps), primalFields, case, homogeneous=False, interprocess=None):
        print case
        rho, beta, sigma = 28. + parameter, 8./3, 10.
        primalFields = [primalFields]
        for i in range(0, nSteps):
            x, y, z = primalFields[-1]
            primalFields.append(primalFields[-1] + self.dt*np.array([
                    sigma*(y-x),
                    x*(rho-z)-y,
                    x*y - beta*z
                ]))
        for i in range(0, nSteps):
            x, y, z = primalFields[nSteps-i]
            ft = np.dot(np.array([[-sigma, sigma, 0],[rho-z,-1,-x],[y,x,-beta]]).T, fields)
            if not homogeneous:
                ft += np.array([0.,0.,1.])
            fields += self.dt*ft
        return fields, 0.

def compute_dxdt_of_order(u, order):
    assert order >= 1
    A = np.array([np.arange(order + 1) ** i for i in range(order + 1)])
    b = np.zeros(order + 1)
    b[1] = 1
    c = np.linalg.solve(A, b)
    return sum([c[i]*u[i] for i in range(0, order+1)])

class NILSAS:
    def __init__(self, (nExponents, nSteps, nSegments, nRuns), (base, time, dt, templates), nProcs, flags=None):
        self.nExponents = nExponents
        self.nSteps = nSteps
        self.nSegments = nSegments
        self.nRuns = nRuns
        self.runner = SerialRunner(base, time, dt, templates, nProcs=nProcs, flags=flags)
        self.primalFields = [self.runner.readFields(base, time)]
        self.nDOF = self.primalFields[0].shape[0]
        self.adjointFields = []
        self.gradientInfo = []
        self.parameter = 0.0
        return

    def initRandom(self):
        W = np.random.randn(self.nExponents, self.nDOF)
        w = np.zeros(self.nDOF)
        self.adjointFields.append((W, w)) 
        return W, w

    def orthogonalize(self, segment, W, w):
        # remove f component
        f = self.getNeutralDirection(segment)
        W = W-np.outer(np.dot(W, f), f)
        w = w-np.dot(w, f)*f

        # QR factorization
        Q, R = np.linalg.qr(W.T)
        W = Q.T
        b = np.dot(W, w)
        w = (w - np.dot(Q, b)).flatten()
        self.gradientInfo.append((R, b))
        self.adjointFields.append((W, w))
        return W, w

    def getNeutralDirection(self, segment):
        orderOfAccuracy = 3
        res = [self.primalFields[segment + 1]]
        for nSteps in range(1, orderOfAccuracy + 1):
            case = self.runner.base + 'segment_{}_neutral_{}_nsteps/'.format(segment, nSteps)
            self.runner.copyCase(case)
            res.append(self.runner.runPrimal(res[0], (self.parameter, nSteps), case)[0])
            self.runner.removeCase(case)
        neutral = compute_dxdt_of_order(res, orderOfAccuracy)
        neutralDirection = neutral/np.linalg.norm(neutral)
        return neutralDirection

    def runPrimal(self):
        # serial process
        for segment in range(len(self.primalFields)-1, self.nSegments):
            case = self.runner.base + 'segment_{}_primal/'.format(segment)
            self.runner.copyCase(case)
            res = self.runner.runPrimal(self.primalFields[segment], (self.parameter, self.nSteps), case)
            self.runner.removeCase(case)
            self.primalFields.append(res[0])
            self.saveCheckpoint()
        return

    def runSegment(self, segment, W, w):
        p = self.primalFields[segment]
        W, w = self.orthogonalize(segment, W, w)
        Wn = []
        Jw = []
        pool = Pool(self.nRuns)
        manager = mp.Manager()
        interprocess = (manager.Lock(), manager.dict())
        #interprocess = None
        segments = []
        def runCase(fields, case, homogeneous, interprocess):
            self.runner.copyCase(case)
            res = self.runner.runAdjoint(fields, (self.parameter, self.nSteps), p, case, homogeneous=homogeneous, interprocess=interprocess)
            #self.runner.removeCase(case)
            return res
        for i in range(0, self.nExponents):
            case = self.runner.base + 'segment_{}_homogeneous_{}/'.format(segment, i)
            segments.append(pool.apply_async(runCase, (W[i], case, True, interprocess)))
        case = self.runner.base + 'segment_{}_inhomogeneous/'.format(segment)
        segments.append(pool.apply_async(runCase, (w, case, False, interprocess)))

        wn, _ = segments[-1].get()
        for i in range(0, self.nExponents):
            res = segments[i].get() 
            Wn.append(res[0])
            Jw.append(res[1])
        Wn = np.array(Wn)
        return Wn, wn

    def computeGradient(self):
        return

    def getExponents(self):
        ii = np.arange(0, self.nExponents)
        exps = []
        for segment in range(0, self.nSegments):
            exp = np.log(np.abs(self.gradientInfo[segment][0][ii,ii]))
            exp /= self.nSteps*self.runner.dt
            exps.append(exp)
            #print exp
        print np.mean(exps[len(exps)/2:], axis=0)

    def saveCheckpoint(self):
        with open(self.runner.base + 'checkpoint.pkl', 'w') as f:
            adjointFields = [None]*len(self.adjointFields)
            adjointFields[-1] = self.adjointFields[-1]
            checkpoint = (self.primalFields, self.adjointFields, self.gradientInfo)
            pickle.dump(checkpoint, f)

    def loadCheckpoint(self):
        checkpointFile = self.runner.base + 'checkpoint.pkl'
        if not os.path.exists(checkpointFile):
            return
        with open(checkpointFile, 'r') as f:
            checkpoint = pickle.load(f)
            self.primalFields, self.adjointFields, self.gradientInfo = checkpoint

    def run(self):
        self.runPrimal()
        if len(self.adjointFields) == 0:
            W, w = self.initRandom()
            self.saveCheckpoint()
        else:
            W, w = self.adjointFields[-1]
        for segment in range(len(self.adjointFields)-1, self.nSegments):
            W, w = self.runSegment(self.nSegments-segment-1, W, w)
            self.saveCheckpoint()
        return 

def main():
    base = '/scratch/talnikar/3d_cylinder/3d_cylinder_adjoint/'
    time = 2.0
    dt = 6e-9
    template = 'templates/3d_cylinder_fds.py'
    nProcs = 16

    nSegments = 100
    nSteps = 500
    nExponents = 20
    nRuns = 4

    # lorenz
    #base = '/home/talnikar/adFVM/cases/3d_cylinder/'
    #time = 10.
    #dt = 0.001
    #nProcs = 1

    #nSegments = 50
    #nSteps = 2000
    ##nSteps = 200
    #nExponents = 2
    ##nExponents = 3
    #nRuns = 2

    runner = NILSAS((nExponents, nSteps, nSegments, nRuns), (base, time, dt, template), nProcs=nProcs, flags=['-g', '--gpu_double'])
    runner.loadCheckpoint()
    runner.run()
    runner.getExponents()

if __name__ == '__main__':
    main()
