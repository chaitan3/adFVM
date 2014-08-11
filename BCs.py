
def zeroGradient(field, indices, patchIndices):
    mesh = field.mesh
    field.field[indices] = field.field[mesh.owner[patchIndices]]

