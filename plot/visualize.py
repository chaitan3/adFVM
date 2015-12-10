import matplotlib.pyplot as plt
import numpy as np

def plot(field, patchID):
    mesh = field.mesh
    m = mesh.origMesh
    p = m.boundary[patchID]
    s = p['startFace']
    e = s + p['nFaces']
    s = m.nInternalCells + s - m.nInternalFaces
    e = m.nInternalCells + e - m.nInternalFaces
    x = m.cellCentres[s:e,1]
    y = m.cellCentres[s:e,2]
    X = x.reshape(50, 50)
    Y = y.reshape(50, 50)

    z = field.field[s:e]
    z[:,0] -= 100
    Z = z.reshape(50, 50, 3)
    Z1 = Z[:,:,1]
    Z2 = Z[:,:,2]
    Z = np.sqrt((Z**2).sum(axis=2))
    inter = 3

    plt.contourf(X, Y, Z, 50)
    plt.colorbar()
    plt.quiver(X[::inter,::inter], Y[::inter,::inter], Z1[::inter,::inter], Z2[::inter,::inter], color='k', linewidth=2)
    plt.show()
