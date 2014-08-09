import numpy as np
import numpad as ad
from field import Field

def interpolate(field):
    mesh = field.mesh
    factor = (mesh.faceDeltas/mesh.deltas).reshape(-1,1)
    faceField = Field.zeros(field.name + 'f', mesh, mesh.nFaces, field.dimensions)
    faceField.field = field.field[mesh.owner]*factor + field.field[mesh.neighbour]*(1-factor)
    # take care of non cyclic boundaries
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
    #mesh = field.mesh
    #return mesh.sumOp.dot(1* mesh.areas)/mesh.volumes).reshape(-1,1)
    # does not work for non cyclic
    return div(interpolate(DT*grad(field), 1.))

def ddt(field, field0, dt):
    return (field.getInternalField()-ad.value(field0.getInternalField()))/dt

def solve(equation, field):
    def solver(internalField):
        field.setInternalField(internalField)
        return equation(field)
    field.setInternalField(ad.solve(solver, field.getInternalField()))

