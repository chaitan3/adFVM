#!/usr/bin/python2
import numpy as np
import os
import shutil
import pickle

from adFVM.interface import SerialRunner

class NILSAS:
    def __init__(self, (nExponents, nSteps, nSegments), (base, time, dt, templates), nProcs, flags=None):
        self.nExponents = nExponents
        self.nSteps = nSteps
        self.nSegments = nSegments
        self.runner = SerialRunner(base, time, dt, template, nProcs=nProcs, flags=flags)
        self.primalFields = [self.runner.readFields(base, time)]
        self.nDOF = self.runner.internalCells.shape[0]
        self.homogeneous = []
        self.inhomogeneous = []
        return

    def initRandom(self):
        W = np.random.rand(self.nExponents, self.nDOF)
        w = np.zeros(self.nDOF)
        return W, w

    def orthogonalNeutral(self, segment, W, w):
        #case = base + 'segment_{}_speed'.format(segment)
        #self.runner.copyCase(case)
        #f = self.runner.runPrimal(self.primalFields[segment + 1], (parameter, 0), case)
	#shutil.rmtree(case)
        W = W-np.dot(W, np.dot(W, f))
        w = w-f*np.dot(w, f)
        Q, R = np.linalg.qr(W.T)
        W = Q.T
        b = np.dot(W, w)
        w = (w - np.dot(Q, b)).flatten()
        return W, w

    def runPrimal(self):
        for i in range(0, self.nSegments):
            res = self.runner.runPrimal(self.primalFields[i], (parameter, self.nSteps))
            self.primalFields.append(res[0])
        return

    def runSegment(self, segment, W, w):
        p = self.primalFields[segment]
        W, w = self.orthogonalNeutral(segment, W, w)
        Wn = []
        Jw = []
        for i in range(0, self.nExponents):
            # homogeneous/inhomogeneous
            case = base + 'segment_{}_homogeneous_{}'.format(segment, i)
            self.runner.copyCase(case)
            res = self.runner.runAdjoint(W[i], (parameter, self.nSteps), p)
            shutil.rmtree(case)
            Wn.append(res[0])
            JW.append(res[1])
        Wn = np.array(Wn)
        case = base + 'segment_{}_inhomogeneous'.format(segment)
        wn, Jw = self.runner.runAdjoint(w, (parameter, nSteps), p)
        shutil.rmtree(case)
        return Wn, wn

    def computeGradient(self):
        return

    def saveCheckpoint(self):
        with open(base + 'checkpoint.pkl') as f:
            checkpoint = (self.primalFields, self.homogeneous, self.inhomogeneous)
            pickle.dump(f, checkpoint)

    def run(self):
        self.runPrimal()
        W, w = self.initRandom()
        self.homogeneous.append(W) 
        self.inhomogeneous.append(w)
        for i in range(0, self.nSegments):
            W, w = self.runSegment(i, W, w)
            self.homogeneous.append(W) 
            self.inhomogeneous.append(w)
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
