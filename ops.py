from __future__ import print_function
import numpy as np
import numpad as ad
import time

from field import Field

def interpolate(field):
    mesh = field.mesh
    factor = (mesh.faceDeltas/mesh.deltas)
    faceField = Field.zeros(field.name + 'f', mesh, mesh.nFaces, field.dimensions)
    faceField.field = field.field[mesh.owner]*factor + field.field[mesh.neighbour]*(1-factor)
    return faceField

def div(field, U):
    mesh = field.mesh
    Fdotn = ad.sum((field.field * U) * mesh.normals, axis=1).reshape((-1,1))
    return (mesh.sumOp * (Fdotn * mesh.areas))/mesh.volumes

def grad(field):
    mesh = field.mesh
    return (mesh.sumOp * (field.field * mesh.normals * mesh.areas))/mesh.volumes

def laplacian(field, DT):
    mesh = field.mesh

    # non orthogonal correction
    DTgradF = Field.zeros('grad' + field.name, mesh, mesh.nCells, 3.)
    DTgradF.setInternalField(DT*grad(field))
    laplacian1 = div(interpolate(DTgradF), 1.)

    gradFdotn = (field.field[mesh.neighbour]-field.field[mesh.owner])/mesh.deltas
    laplacian2 = (mesh.sumOp * (DT * gradFdotn * mesh.areas))/mesh.volumes
    return laplacian2

def ddt(field, field0, dt):
    return (field.getInternalField()-ad.value(field0.getInternalField()))/dt

def solve(equation, field):
    start = time.time()
    
    def solver(internalField):
        field.setInternalField(internalField)
        return equation(field)
    field.setInternalField(ad.solve(solver, field.getInternalField()))

    end = time.time()
    print('Time for iteration:', end-start)

