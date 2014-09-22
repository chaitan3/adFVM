import numpy as np

from field import CellField

from utils import ad, pprint
from utils import Logger
logger = Logger(__name__)
import time

def euler(equation, boundary, fields, solver):
    start = time.time()

    names = [phi.name for phi in fields]
    pprint('Time marching for', ' '.join(names))
    for index in range(0, len(fields)):
        fields[index].old = fields[index]
        fields[index].info()
    LHS = equation(*fields)
    internalFields = [(fields[index].getInternalField() - LHS[index].field*solver.dt) for index in range(0, len(fields))]
    newFields = boundary(*internalFields)
    for index in range(0, len(fields)):
        newFields[index].name = fields[index].name

    end = time.time()
    pprint('Time for iteration:', end-start)
    return newFields

def RK(equation, boundary, fields, solver):
    start = time.time()

    names = [phi.name for phi in fields]
    pprint('Time marching for', ' '.join(names))
    for phi in fields:
        phi.info()

    def NewFields(a, LHS):
        internalFields = [phi.getInternalField().copy() for phi in fields]
        for termIndex in range(0, len(a)):
            for index in range(0, len(fields)):
                internalFields[index] -= a[termIndex]*LHS[termIndex][index].field*solver.dt
        return boundary(*internalFields)

    def f(a, *LHS):
        pprint('RK step ', f.rk)
        f.rk += 1
        if len(LHS) != 0:
            newFields = NewFields(a, LHS)
        else:
            newFields = fields
        for phi in newFields:
            phi.old = phi
        return equation(*newFields)
    f.rk = 1

    k1 = f([0.])
    k2 = f([0.5], k1)
    k3 = f([0.5], k2)
    k4 = f([1.], k3)
    newFields = NewFields([1./6, 1./3, 1./3, 1./6], [k1, k2, k3, k4])

    for index in range(0, len(fields)):
        newFields[index].name = fields[index].name
    end = time.time()
    pprint('Time for iteration:', end-start)
    return newFields



def implicit(equation, boundary, fields, garbage):
    assert ad.__name__ == 'numpad'
    start = time.time()

    names = [phi.name for phi in fields]
    pprint('Solving for', ' '.join(names))
    for index in range(0, len(fields)):
        fields[index].old = CellField.copy(fields[index])
        fields[index].info()
    nDimensions = np.concatenate(([0], np.cumsum(np.array([phi.dimensions[0] for phi in fields]))))
    nDimensions = zip(nDimensions[:-1], nDimensions[1:])
    def setInternalFields(stackedInternalFields):
        internalFields = []
        # range creates a copy on the array
        for index in range(0, len(fields)):
            internalFields.append(stackedInternalFields[:, range(*nDimensions[index])])
        return boundary(*internalFields)
    def solver(internalFields):
        newFields = setInternalFields(internalFields)
        for index in range(0, len(fields)):
            newFields[index].old = fields[index].old
        return ad.hstack([phi.field for phi in equation(*newFields)])

    internalFields = ad.hstack([phi.getInternalField() for phi in fields])
    solution = ad.solve(solver, internalFields)
    newFields = setInternalFields(solution)
    for index in range(0, len(fields)):
        newFields[index].name = fields[index].name

    end = time.time()
    pprint('Time for iteration:', end-start)
    return newFields

def derivative(newField, oldFields):
    names = [phi.name for phi in oldFields]
    logger.info('computing derivative with respect to {0}'.format(names))
    diffs = [newField.diff(phi.field).toarray().reshape(phi.field.shape) for phi in oldFields]
    result = np.hstack(diffs)
    return result

def copy(fields):
    newFields = [CellField.copy(phi) for phi in fields]
    return newFields

def forget(fields):
    if ad.__name__ != 'numpad':
        return
    for phi in fields:
        phi.field.obliviate()




