from __future__ import print_function
import numpy as np
import numpad as ad

def zeroGradient(field, indices, patchIndices):
    mesh = field.mesh
    field.field[indices] = field.field[mesh.owner[patchIndices]]

