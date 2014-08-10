import numpy as np
import numpad as ad
from field import Field

def interpolate(field):
    mesh = field.mesh
    factor = (mesh.faceDeltas/mesh.deltas).reshape(-1,1)
    faceField = Field.zeros(field.name + 'f', mesh, mesh.nFaces, field.dimensions)
    faceField.field = field.field[mesh.owner]*factor + field.field[mesh.neighbour]*(1-factor)
    return faceField

def div(field, U):
    mesh = field.mesh
    Fdotn = ad.sum((field.field * U) * mesh.normals, axis=1)
    return ((mesh.sumOp * (Fdotn * mesh.areas))/mesh.volumes).reshape((-1,1))

def grad(field):
    mesh = field.mesh
    return (mesh.sumOp * (field.field * mesh.normals * mesh.areas.reshape((-1,1))))/mesh.volumes.reshape((-1,1))

def laplacian(field, DT):
    # non orthogonal correction
    # does not work for non cyclic
    mesh = field.mesh

    DTgradF = Field.zeros('grad' + field.name, mesh, mesh.nCells, 3.)
    DTgradF.setInternalField(DT*grad(field))
    laplacian1 = div(interpolate(DTgradF), 1.)

    gradFdotn = (field.field[mesh.neighbour]-field.field[mesh.owner])/mesh.deltas.reshape(-1,1)
    laplacian2 = (mesh.sumOp * (DT * gradFdotn * mesh.areas.reshape(-1,1)))/mesh.volumes.reshape(-1,1)
    return laplacian2

def ddt(field, field0, dt):
    return (field.getInternalField()-ad.value(field0.getInternalField()))/dt

def solve(equation, field):
    def solver(internalField):
        field.setInternalField(internalField)
        return equation(field)
    field.setInternalField(ad.solve(solver, field.getInternalField()))

