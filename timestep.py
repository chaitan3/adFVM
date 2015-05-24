import numpy as np

from config import ad
from field import Field, CellField

createFields = lambda internalFields, solver : [Field(solver.names[index], phi, solver.dimensions[index]) for index, phi in enumerate(internalFields)]

def euler(equation, boundary, stackedFields, solver):
    solver.stage = 1
    paddedStackedFields = solver.padField(stackedFields)
    paddedFields = solver.unstackFields(paddedStackedFields, CellField)
    LHS = equation(*paddedFields)
    internalFields = [(paddedFields[index].getInternalField() - LHS[index].field*solver.dt) for index in range(0, len(paddedFields))]
    internalFields = createFields(internalFields, solver)
    newFields = boundary(*internalFields)
    return solver.stackFields(newFields, ad)

# classical
def RK2(equation, boundary, stackedFields, solver):
    solver.stage = 1
    paddedStackedFields0 = solver.padField(stackedFields)
    paddedFields0 = solver.unstackFields(paddedStackedFields0, CellField)
    LHS = equation(*paddedFields0)
    internalFields = [(paddedFields0[index].getInternalField() - LHS[index].field*solver.dt/2) for index in range(0, len(paddedFields0))]
    internalFields = createFields(internalFields, solver)
    fields1 = boundary(*internalFields)

    solver.stage = 2
    paddedStackedFields1 = solver.padField(solver.stackFields(fields1, ad))
    paddedFields1 = solver.unstackFields(paddedStackedFields1, CellField)
    LHS = equation(*paddedFields1)
    internalFields = [(paddedFields0[index].getInternalField() - LHS[index].field*solver.dt) for index in range(0, len(paddedFields0))]

    internalFields = createFields(internalFields, solver)
    newFields = boundary(*internalFields)
    return solver.stackFields(newFields, ad)

# classical
def RK4(equation, boundary, stackedFields, solver):
    fields = solver.unstackFields(stackedFields, CellField)
    def NewFields(a, LHS):
        internalFields = [phi.getInternalField() for phi in fields]
        for termIndex in range(0, len(a)):
            for index in range(0, len(fields)):
                internalFields[index] -= a[termIndex]*LHS[termIndex][index].field*solver.dt
        internalFields = createFields(internalFields, solver)
        return boundary(*internalFields)

    def f(a, *LHS):
        if len(LHS) != 0:
            newFields = NewFields(a, LHS)
        else:
            newFields = fields
        paddedStackedFields = solver.padField(solver.stackFields(newFields, ad))
        paddedFields = solver.unstackFields(paddedStackedFields, CellField)
        return equation(*paddedFields)

    solver.stage = 1
    k1 = f([0.])
    solver.stage = 2
    k2 = f([0.5], k1)
    solver.stage = 3
    k3 = f([0.5], k2)
    solver.stage = 4
    k4 = f([1.], k3)
    newFields = NewFields([1./6, 1./3, 1./3, 1./6], [k1, k2, k3, k4])

    return solver.stackFields(newFields, ad)

def SSPRK(equation, boundary, stackedFields, solver):
    # 2nd order
    #alpha = np.array([[1.,0],[1./2,1./2]])
    #beta = np.array([[1,0],[0,1./2]])
    # 3rd order
    alpha = np.array([[1,0,0],[3./4,1./4, 0],[1./3,0,2./3]])
    beta = np.array([[1,0,0],[0,1./4,0],[0,0,2./3]])
    nStages = alpha.shape[0]
    LHS = []
    fields = []
    fields.append(solver.unstackFields(stackedFields, CellField))
    nFields = len(fields[0])
    for i in range(0, nStages):
        solver.stage += 1
        paddedStackedFields = solver.padField(solver.stackFields(fields[i], ad))
        paddedFields = solver.unstackFields(paddedStackedFields, CellField)
        LHS.append(equation(*paddedFields))
        internalFields = [0]*nFields
        for j in range(0, i+1):
            for index in range(0, nFields):
                internalFields[index] += alpha[i,j]*fields[j][index].getInternalField()-beta[i,j]*LHS[j][index].field*solver.dt
        internalFields = createFields(internalFields, solver)
        fields.append(boundary(*internalFields))
        
    return solver.stackFields(fields[-1], ad)

# DOES NOT WORK
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

