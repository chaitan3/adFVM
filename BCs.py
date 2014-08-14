from __future__ import print_function
import numpy as np
import numpad as ad

import utils
logger = utils.logger(__name__)

def cyclic(field, patch, indices, patchIndices):
    logger.debug('cyclic BC for {0}'.format(patch))
    mesh = field.mesh
    nFaces = mesh.boundary[patch]['nFaces']
    neighbourPatch = mesh.boundary[mesh.boundary[patch]['neighbourPatch']]   
    neighbourStartFace = neighbourPatch['startFace']
    neighbourEndFace = neighbourStartFace + nFaces
    field.field[indices] = field.field[mesh.owner[neighbourStartFace:neighbourEndFace]]

def zeroGradient(field, patch, indices, patchIndices):
    logger.debug('zeroGradient BC for {0}'.format(patch))
    mesh = field.mesh
    field.field[indices] = field.field[mesh.owner[patchIndices]]

def symmetryPlane(field, patch, indices, patchIndices):
    logger.debug('symmetryPlane BC for {0}'.format(patch))
    mesh = field.mesh
    zeroGradient(field, patch, indices, patchIndices)
    if field.field.shape[1] == 3:
        v = -mesh.normals[patchIndices]
        field.field[indices] -= 2*ad.sum(field.field[indices]*v, axis=1).reshape((-1,1))*v

def fixedValue(field, patch, indices, patchIndices):
    logger.debug('fixedValue BC for {0}'.format(patch))
    field.field[indices] = field.boundary[patch]['Rvalue']

slip = symmetryPlane
empty = zeroGradient
inletOutlet = zeroGradient
    

