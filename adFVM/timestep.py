import numpy as np

from . import config
from .field import Field
from adpy.tensor import Kernel

createFields = lambda internalFields, solver : [Field(solver.names[index], phi, solver.dimensions[index]) for index, phi in enumerate(internalFields)]

def euler():
    alpha = np.array([[1.]], config.precision)
    beta = np.array([[1.]], config.precision)
    gamma = np.array([0.], config.precision)
    return [alpha, beta, gamma]

# put rk2 and rk4

def SSPRK():
    # 2nd order
    #alpha = np.array([[1.,0],[1./2,1./2]])
    #beta = np.array([[1,0],[0,1./2]])
    # 3rd order
    alpha = np.array([[1,0,0],[3./4,1./4, 0],[1./3,0,2./3]], config.precision)
    beta = np.array([[1,0,0],[0,1./4,0],[0,0,2./3]], config.precision)
    gamma = np.array([0.,1,0.5], config.precision)
    return [alpha, beta, gamma]

def timeStepper(equation, initFields, solver):
    mesh = solver.mesh.symMesh
    alpha, beta, gamma = solver.timeStepCoeff
    nStages = alpha.shape[0]
    LHS = []
    fields = [initFields]
    n = len(fields[0])

    def update(*args, **kwargs):
        i = kwargs['i']
        currFields = [0]*n
        LHS, S, fields, dt = args[:n], args[n:2*n], args[2*n:-1], args[-1]
        dt = dt.scalar()
        for j in range(0, i+1):
            for index in range(0, n):
                currFields[index] += alpha[i,j]*fields[j*n+index]
        for index in range(0, n):
            currFields[index] += -beta[i,i]*(LHS[index]-S[index])*dt
        return tuple(currFields)

    for i in range(0, nStages):
        #solver.t = solver.t0 + gamma[i]*solver.dt
        LHS = equation(*fields[i])
        S = [x[0] for x in solver.sourceTerms]
        args = list(LHS) + S + sum(fields, []) + [solver.dt]
        currFields = Kernel(update)(mesh.nInternalCells)(*args, i=i)
        solver.stage += 1
        fields.append(list(currFields))
    return fields[-1]

