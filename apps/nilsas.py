#!/usr/bin/python2
import numpy as np
import os
import shutil

from adFVM.interface import SerialRunner

class NILSAS:
    def __init__(self, (nExponents, nSteps, nSegments), (base, time, dt, templates), nProcs, flags=None):
        self.nExponents = nExponents
        self.nSteps = nSteps
        self.nSegments = nSegments
        self.runner = SerialRunner(base, time, dt, template, nProcs=nProcs, flags=flags)
        self.primalFields = [self.runner.readFields(base, time)]
        self.nDOF = self.runner.internalCells.shape[0]
        return

    def initRandom(self):
        W = np.random.rand(self.nExponents, self.nDOF)
        w = np.zeros(self.nDOF)
        return W, w

    def orthogonalize(self, W, w):
        return W, w

    def runPrimal(self):
        for i in range(0, self.nSegments):
            res = self.runner.runPrimal(self.primalFields[i], (parameter, self.nSteps))
            self.primalFields.append(res[0])
        return

    def runSegment(self, segment, W, w):
        p = self.primalFields[segment]
        Wn = []
        Jw = []
        for i in range(0, self.nExponents):
            res = self.runner.runAdjoint(W[i], (parameter, self.nSteps), p)
            Wn.append(res[0])
            JW.append(res[1])
        Wn = np.array(Wn)
        wn, Jw = self.runner.runAdjoint(w, (parameter, nSteps), p)
        return Wn, wn

    def computeGradient(self):
        return

    def run(self):
        self.runPrimal()
        W, w = self.initRandom()
        for i in range(0, self.nSegments):
            W, w = self.orthogonalize(W, w)
            self.runSegment(i, W, w)
        return

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
