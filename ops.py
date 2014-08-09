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
    UdotFn = ad.sum((field.field * U.field) * mesh.normals, axis=1)
    return (mesh.sumOp.dot(UdotFn * mesh.areas)/mesh.volumes).reshape(-1,1)

def ddt(field, dt):
    internalField = field.getInternalField()
    return (internalField-ad.base(internalField))/dt

def solve(equation, field):
    def solver(internalField):
        field.setInternalField(internalField)
        return equation(field)
    field.setInternalField(ad.solve(solver, field.getInternalField()))

