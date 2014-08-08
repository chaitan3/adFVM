import numpy as np
from field import FaceField

def interpolate(field):
    mesh = field.mesh
    factor = (mesh.faceDeltas/mesh.deltas).reshape(-1,1)
    faceField = FaceField.zeros(field.name, mesh, field.dimensions)
    faceField.field = factor*field.field[mesh.owner] + (1-factor)*field.field[mesh.neighbour]
    # take care of non cyclic boundaries
    return faceField

def div(field, U):
    mesh = field.mesh
    UdotFn = np.sum((U.field * field.field) * mesh.normals, axis=1)
    return (mesh.sumOp.dot(UdotFn * mesh.areas)/mesh.volumes).reshape(-1,1)

