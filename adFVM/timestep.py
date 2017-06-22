import numpy as np

from . import config
from .field import Field

createFields = lambda internalFields, solver : [Field(solver.names[index], phi, solver.dimensions[index]) for index, phi in enumerate(internalFields)]

#def euler(equation, boundary, stackedFields, solver):
#    solver.stage = 1
#    fields = solver.unstackFields(stackedFields, CellField)
#    LHS = equation(*fields)
#    internalFields = [(fields[index].getInternalField() - LHS[index].field*solver.dt) for index in range(0, len(fields))]
#    internalFields = createFields(internalFields, solver)
#    newFields = boundary(*internalFields)
#    return solver.stackFields(newFields, ad)

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
    alpha, beta, gamma = solver.timeStepCoeff
    nStages = alpha.shape[0]
    LHS = []
    fields = [initFields]
    nFields = len(fields[0])
    for i in range(0, nStages):
        solver.t = solver.t0 + gamma[i]*solver.dt
        LHS.append(equation(*fields[i]))
        internalFields = [0]*nFields
        for j in range(0, i+1):
            for index in range(0, nFields):
                internalFields[index] += alpha[i,j]*fields[j][index].field-beta[i,j]*LHS[j][index].field*solver.dt
        internalFields = createFields(internalFields, solver)
        solver.stage += 1
        if i == nStages-1:
            return  [phi.field for phi in internalFields]
        else:
            fields.append(internalFields)

# DOES NOT WORK
#def implicit(equation, boundary, fields, garbage):
#    assert ad.__name__ == 'numpad'
#    start = time.time()
#
#    names = [phi.name for phi in fields]
#    pprint('Solving for', ' '.join(names))
#    for index in range(0, len(fields)):
#        fields[index].old = CellField.copy(fields[index])
#        fields[index].info()
#    nDimensions = np.concatenate(([0], np.cumsum(np.array([phi.dimensions[0] for phi in fields]))))
#    nDimensions = zip(nDimensions[:-1], nDimensions[1:])
#    def setInternalFields(stackedInternalFields):
#        internalFields = []
#        # range creates a copy on the array
#        for index in range(0, len(fields)):
#            internalFields.append(stackedInternalFields[:, range(*nDimensions[index])])
#        return boundary(*internalFields)
#    def solver(internalFields):
#        newFields = setInternalFields(internalFields)
#        for index in range(0, len(fields)):
#            newFields[index].old = fields[index].old
#        return ad.hstack([phi.field for phi in equation(*newFields)])
#
#    internalFields = ad.hstack([phi.getInternalField() for phi in fields])
#    solution = ad.solve(solver, internalFields)
#    newFields = setInternalFields(solution)
#    for index in range(0, len(fields)):
#        newFields[index].name = fields[index].name
#
#    end = time.time()
#    pprint('Time for iteration:', end-start)
#    return newFields

