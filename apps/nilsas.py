#!/usr/bin/python2
import numpy as np
import os
import shutil
import pickle

from adFVM.interface import SerialRunner

def compute_dxdt_of_order(u, order):
    assert order >= 1
    A = np.array([np.arange(order + 1) ** i for i in range(order + 1)])
    b = np.zeros(order + 1)
    b[1] = 1
    c = np.linalg.solve(A, b)
    return sum([c[i]*u[i] for i in range(0, order+1)])

class NILSAS:
    def __init__(self, (nExponents, nSteps, nSegments), (base, time, dt, templates), nProcs, flags=None):
        self.nExponents = nExponents
        self.nSteps = nSteps
        self.nSegments = nSegments
        self.runner = SerialRunner(base, time, dt, template, nProcs=nProcs, flags=flags)
        self.nDOF = self.runner.internalCells.shape[0]
        self.primalFields = [self.runner.readFields(base, time)]
        self.adjointFields = []
        self.gradientInfo = []
        return

    def initRandom(self):
        W = np.random.rand(self.nExponents, self.nDOF)
        w = np.zeros(self.nDOF)
        self.adjointFields.append((W, w)) 
        return W, w

    def orthogonalize(self, segment, W, w):
        f = self.getNeutralDirection(segment)
        # remove f component
        W = W-np.dot(W, np.dot(W, f))
        w = w-f*np.dot(w, f)

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
            case = base + 'neutral_{}_{}_nsteps'.format(segment, nSteps)
            self.runner.copyCase(case)
            res.append(self.runner.runPrimal(res[0], (parameter, nSteps))[0])
            shutil.rmtree(case)
        neutral = compute_dxdt_of_order(res, orderOfAccuracy)
        neutralDirection /= np.linalg.norm(neutral, axis=0)
        return neutralDirection

    def runPrimal(self):
        for segment in range(len(self.primalFields)-1, self.nSegments):
            res = self.runner.runPrimal(self.primalFields[segment], (parameter, self.nSteps))
            self.primalFields.append(res[0])
            self.saveCheckpoint()
        return

    def runSegment(self, segment, W, w):
        p = self.primalFields[segment]
        W, w = self.orthogonalize(segment, W, w)
        Wn = []
        Jw = []
        for i in range(0, self.nExponents):
            # homogeneous/inhomogeneous
            case = base + 'segment_{}_homogeneous_{}'.format(segment, i)
            self.runner.copyCase(case)
            res = self.runner.runAdjoint(W[i], (parameter, self.nSteps), p, homogeneous=True)
            shutil.rmtree(case)
            Wn.append(res[0])
            Jw.append(res[1])
        Wn = np.array(Wn)
        case = base + 'segment_{}_inhomogeneous'.format(segment)
        wn, _ = self.runner.runAdjoint(w, (parameter, nSteps), p)
        shutil.rmtree(case)
        return Wn, wn

    def computeGradient(self):
        return

    def saveCheckpoint(self):
        with open(base + 'checkpoint.pkl', 'w') as f:
            checkpoint = (self.primalFields, self.adjointFields, self.gradientInfo)
            pickle.dump(f, checkpoint)

    def loadCheckpoint(self):
        with open(base + 'checkpoint.pkl', 'r') as f:
            checkpoint = pickle.load(f)
            self.primalFields, self.adjointFields, self.gradientInfo = checkpoint

    def run(self):
        self.runPrimal()
        if len(self.adjointFields) == 0:
            W, w = self.initRandom()
            self.saveCheckpoint()
        for segment in range(len(self.adjointFields)-1, self.nSegments):
            W, w = self.runSegment(segment, W, w)
            self.saveCheckpoint()
        return 'booya'

def main():
    base = 'cases/3d_cylinder/'
    time = 2.0
    dt = 6e-9
    template = 'templates/3d_cylinder_fds.py'
    nProcs = 1

    nSegments = 400
    nSteps = 500
    nExponents = 20
    parameter = 0.0

    runner = NILSAS((nExponents, nSteps, nSegments), (base, time, dt, template), nProcs=nProcs, flags=['-g', '--gpu_double'])

if __name__ == '__main__':
    main()
