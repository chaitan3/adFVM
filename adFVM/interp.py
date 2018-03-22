from . import config, parallel
from .field import Field#, faceExchange

import itertools
import numpy as np
from scipy import sparse as sparse

logger = config.Logger(__name__)

# central is second order with no grad
# dual should do second order with grad
# characteristic is extrapolated

# dual defined for np and ad
def central(phi, mesh):
    f = mesh.weights
    phiF = phi.extract(mesh.owner)*f + phi.extract(mesh.neighbour)*(-f+1)
    return phiF

def secondOrder(phi, gradPhi, mesh, swap):
    p, n = mesh.owner, mesh.neighbour
    if swap:
        n, p = p, n
    phiC, phiD = phi.extract(p), phi.extract(n)
    phiF = phiC + (phiD-phiC)*mesh.linearWeights[swap] 
    for i in range(0, phiC.shape[0]):
        phiF[i] += mesh.quadraticWeights[swap].dot(gradPhi.extract(p)[i])
    return phiF

def centralOld(phi, mesh):
    logger.info('interpolating {0}'.format(phi.name))
    factor = mesh.weights
    # for tensor
    if len(phi.dimensions) == 2:
        factor = np.reshape(factor, (-1, 1, 1))
    faceField = Field('{0}F'.format(phi.name), phi.field[mesh.owner]*factor + phi.field[mesh.neighbour]*(1.-factor), phi.dimensions)
    #faceField = Field('{0}F'.format(phi.name), phi.field[mesh.owner], phi.dimensions)
    # retain pattern broadcasting
    #if hasattr(mesh, 'origMesh'):
    #    faceField.field = ad.patternbroadcast(faceField.field, phi.field.broadcastable)
    return faceField

