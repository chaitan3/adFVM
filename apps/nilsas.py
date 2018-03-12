#!/usr/bin/python2
import numpy as np
from scipy import sparse
import scipy.sparse.linalg as splinalg
import os
import shutil
import pickle
from pathos.multiprocessing import Pool
from pathos.helpers import mp

from adFVM.interface import SerialRunner
class SerialRunner(object):
#class SerialRunnerLorenz(object):
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
        rho, beta, sigma = 28. + parameter, 8./3, 10.
        for i in range(0, nSteps):
            x, y, z = fields
            fields = fields + self.dt*np.array([
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
        vfs = []
        for i in range(0, nSteps):
            x, y, z = primalFields[nSteps-i]
            ft = np.dot(np.array([[-sigma, sigma, 0],[rho-z,-1,-x],[y,x,-beta]]).T, fields)
            if not homogeneous:
                ft = ft + np.array([0.,0.,1.])
            fs = np.array([0., x, 0.])

            vfs.append(np.dot(fields, fs))
            fields = fields + self.dt*ft
        return fields, np.array(vfs)

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
        self.lssInfo = []
        self.sensitivities = []
        self.parameter = 0.0
        self.checkpointInterval = 1000
        return

    def initRandom(self):
        W = np.random.randn(self.nExponents, self.nDOF)
        w = np.zeros(self.nDOF)
        return W, w

    # forward index
    def orthogonalize(self, segment, W, w, neutral=True):
        # remove f component
        #if neutral:
        if len(self.gradientInfo) == 0:
            f = self.getNeutralDirection(segment)
            W = W-np.outer(np.dot(W, f), f)
            w = w-np.dot(w, f)*f

        # QR factorization
        Q, R = np.linalg.qr(W.T)
        W = Q.T
        b = np.dot(W, w)
        w = w - np.dot(Q, b)
        self.gradientInfo.append((R, b))
        return W, w

    # forward index
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

    # forward index
    def runPrimal(self):
        # serial process
        for segment in range(len(self.primalFields)-1, self.nSegments):
            case = self.runner.base + 'segment_{}_primal/'.format(segment)
            self.runner.copyCase(case)
            res = self.runner.runPrimal(self.primalFields[segment], (self.parameter, self.nSteps), case)
            self.runner.removeCase(case)
            self.primalFields.append(res[0])
            if (segment + 1) % self.checkpointInterval == 0:
                self.saveCheckpoint()
        return

    # forward index
    def runSegment(self, segment, W, w):
        primalFields = self.primalFields[segment]
        W, w = self.orthogonalize(segment, W, w)
        Wn = []
        def runCase(runner, fields, primalData, primalFields, case, homogeneous, interprocess):
            runner.copyCase(case)
            res = runner.runAdjoint(fields, primalData, primalFields, case, homogeneous=homogeneous, interprocess=interprocess)
            runner.removeCase(case)
            return res

        interprocess = None

        #pool = Pool(self.nRuns)
        #manager = mp.Manager()
        #interprocess = (manager.Lock(), manager.dict())
        #segments = []
        #for i in range(0, self.nExponents):
        #    case = self.runner.base + 'segment_{}_homogeneous_{}/'.format(segment, i)
        #    segments.append(pool.apply_async(runCase, (self.runner, W[i], (self.parameter, self.nSteps), primalFields, case, True, interprocess)))
        #case = self.runner.base + 'segment_{}_inhomogeneous/'.format(segment)
        #segments.append(pool.apply_async(runCase, (self.runner, w, (self.parameter, self.nSteps), primalFields, case, False, interprocess)))

        #JW = []
        #for i in range(0, self.nExponents):
        #    res = segments[i].get() 
        #    Wn.append(res[0])
        #    JW.append(res[1])
        #JW = np.array(JW)
        #wn, Jw = segments[-1].get()

        JW = []
	for i in range(0, self.nExponents):
            case = self.runner.base + 'segment_{}_homogeneous_{}/'.format(segment, i)
            res = runCase(self.runner, W[i], (self.parameter, self.nSteps), primalFields, case, True, interprocess)
            Wn.append(res[0])
            JW.append(res[1])
        JW = np.array(JW)
        case = self.runner.base + 'segment_{}_inhomogeneous/'.format(segment)
        wn, Jw = runCase(self.runner, w, (self.parameter, self.nSteps), primalFields, case, False, interprocess)

        self.sensitivities.append((JW, Jw))
        Wn = np.array(Wn)

        #self.lssInfo.append((np.dot(Wn, Wn.T), np.dot(Wn, wn)))
        return Wn, wn

    def solveLSS(self):
        Rs = [x[0] for x in self.gradientInfo[1:]]
        bs = [x[1] for x in self.gradientInfo[1:]]
        R, b = np.array(Rs), np.array(bs)
        assert R.ndim == 3 and b.ndim == 2
        assert R.shape[0] == b.shape[0]
        assert R.shape[1] == R.shape[2] == b.shape[1]
        nseg, subdim = b.shape
        eyes = np.eye(subdim, subdim) * np.ones([nseg, 1, 1])
        matrix_shape = (subdim * nseg, subdim * (nseg+1))
        I = sparse.bsr_matrix((eyes, np.r_[1:nseg+1], np.r_[:nseg+1]))
        D = sparse.bsr_matrix((R, np.r_[:nseg], np.r_[:nseg+1]), shape=matrix_shape)
        B = (D - I).tocsr()
        Schur = B * B.T #+ 1E-5 * sparse.eye(B.shape[0])
        alpha = -(B.T * splinalg.spsolve(Schur, np.ravel(b)))
        # alpha1 = splinalg.lsqr(B, ravel(bs), iter_lim=10000)
        coeff = alpha.reshape([nseg+1,-1])[:-1]
        return coeff

    def computeGradient(self, coeff):
        N = len(self.sensitivities)
        assert coeff.shape == (N, self.nExponents)
        assert len(coeff) == N
        ws = sum([x[1].sum()/self.nSteps for x in self.sensitivities])/N
        Ws = sum([np.dot(x[0].T, coeff[i]).sum()/self.nSteps for i, x in enumerate(self.sensitivities)])/N
        return ws + Ws

    # reverse index
    def getExponents(self, start=None):
        ii = np.arange(0, self.nExponents)
        exps = []
        for segment in range(1, len(self.gradientInfo)):
            exp = np.log(np.abs(self.gradientInfo[segment][0][ii,ii]))
            exp /= self.nSteps*self.runner.dt
            exps.append(exp)
            #print exp
        if start is None:
            start = len(exps)/2
        return np.mean(exps[start:], axis=0), np.std(exps[start:], axis=0)/np.sqrt(len(exps)-start)

    def saveVectors(self):
        for i in range(0, self.nExponents):
            self.runner.writeFields(self.adjointFields[-1][0][i], self.runner.base, 10.0 + i, adjoint=True)
        return

    def saveCheckpoint(self):
        checkpointFile = self.runner.base + 'checkpoint_temp.pkl'
        with open(checkpointFile, 'w') as f:
            checkpoint = (self.primalFields, self.adjointFields, self.gradientInfo, self.lssInfo, self.sensitivities)
            pickle.dump(checkpoint, f)
        shutil.move(checkpointFile, self.runner.base + 'checkpoint.pkl')

    def loadCheckpoint(self):
        checkpointFile = self.runner.base + 'checkpoint.pkl'
        if not os.path.exists(checkpointFile):
            return
        with open(checkpointFile, 'r') as f:
            checkpoint = pickle.load(f)
            self.primalFields, self.adjointFields, self.gradientInfo, self.lssInfo, self.sensitivities = checkpoint

    # reverse index
    def run(self):
        self.runPrimal()
        if len(self.adjointFields) == 0:
            W, w = self.initRandom()
            self.adjointFields.append((W, w))
            self.saveCheckpoint()
        else:
            W, w = self.adjointFields[-1]
        for segment in range(len(self.adjointFields)-1, self.nSegments):
            W, w = self.runSegment(self.nSegments-segment-1, W, w)
            self.adjointFields[-1] = None
            self.adjointFields.append((W, w))
            if (segment + 1) == self.nSegments:
                self.orthogonalize(-1, W, w, neutral=False)
            if (segment + 1) % self.checkpointInterval == 0:
                self.saveCheckpoint()
        return 

def main():
    base = '/home/talnikar/adFVM/cases/3d_cylinder/nilsas/endeavour/'
    time = 2.0
    dt = 6e-9
    template = 'templates/3d_cylinder_fds.py'
    nProcs = 1

    nSegments = 100
    nSteps = 500
    nExponents = 20
    nRuns = 1

    # lorenz
    base = '/home/talnikar/adFVM/cases/3d_cylinder/'
    time = 10.
    dt = 0.0001
    nProcs = 1

    nSegments = 100
    nSteps = 4000
    #nSteps = 200
    #nExponents = 2
    nExponents = 2
    nRuns = 1

    runner = NILSAS((nExponents, nSteps, nSegments, nRuns), (base, time, dt, template), nProcs=nProcs, flags=['-g', '--gpu_double'])
    #runner.loadCheckpoint()
    runner.run()
    print 'exponents', runner.getExponents()
    ###coeff = np.ones((nSegments, nExponents))
    #coeff = runner.solveLSS()
    #print 'gradient', runner.computeGradient(coeff)
    #runner.saveVectors()
    import pdb;pdb.set_trace()

if __name__ == '__main__':
    main()
