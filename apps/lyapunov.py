#!/usr/bin/python

import numpy as np
import time
import sys

from problem import primal, nSteps, writeInterval, nExponents
from adjoint import Adjoint


def readFields():
    return fields

def writeFields(fields):
    return

def main():
    nIntervals = nSteps/writeInterval
    adjoint = Adjoint(primal)

    primal.readFields(startTime)
    adjoint.createFields()
    adjoint.compile()

    data = adjoint.initPrimalData()
    adjoint.forceReadFields = user.readFields

    adjoint.run(*data)
    for interval in range(0, nIntervals):
        stackedAdjointFields = parallel.gatherCells(stackedAdjointFields, mesh.origMesh, axis=1)
        if stackedAdjointFields is not None:
            totalCells = stackedAdjointFields.shape[1]
            A = stackedAdjointFields.reshape((nLyapunov, totalCells*nDims))
            Q, R = np.linalg.qr(A.T)
            S = np.diag(np.sign(np.diag(R)))
            Q, R = np.dot(Q, S), np.dot(S, R)
            r = np.log(np.diag(R))/(t0-t)
            exponents.append(r)
            stackedAdjointFields = Q.T.reshape((nLyapunov, totalCells, nDims))
        stackedAdjointFields = parallel.scatterCells(stackedAdjointFields, mesh.origMesh, axis=1)

        for sim in range(0, nExponents):
            names = ['{0}a_{1}'.format(name, sim) for name in primal.names]
            writeAdjointFields(stackedAdjointFields[sim], names, t)

    exponents = np.array(exponents)
    pprint(exponents)
    pprint(np.mean(exponents, axis=0))

if __name__ == '__main__':
    main()
