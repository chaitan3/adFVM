from __future__ import print_function
from field import CellField

from utils import ad, pprint
from utils import Logger
logger = Logger(__name__)
import time

def explicit(equation, boundary, fields, solver):
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
        #newFields[index].old = fields[index]

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




