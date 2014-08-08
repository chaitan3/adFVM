import numpy as np

def interpolate(field, mesh):
    factor = (mesh.faceDeltas/mesh.deltas).reshape(-1,1)
    return factor*field[mesh.owner[:mesh.nInternalFaces]] + (1-factor)*field[mesh.neighbour]

def div(field, boundary, U, mesh):
    fieldAndBoundary = np.concatenate((field, boundary))
    UdotFn = np.sum((U * fieldAndBoundary) * mesh.normals, axis=1)
    return (mesh.sumOp.dot(UdotFn * mesh.areas)/mesh.volumes).reshape(-1,1)

#put both cyclics in internal cells
#pass extended field to interpolate
#update ghost fields at the end of time integration
#separate field object might be necessary
#mesh read boundary
