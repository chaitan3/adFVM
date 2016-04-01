cimport numpy as np
cimport cython
ctypedef double dtype

import numpy as np

def add_at(np.ndarray[dtype, ndim=2] a, np.ndarray[int] indices, np.ndarray[dtype,ndim=2] b):
    cdef int n = indices.shape[0]
    cdef int m = a.shape[1]
    for i in range(0, n):
        for j in range(0, m):
            a[indices[i], j] += b[i, j]

@cython.boundscheck(False)
def intersectPlane(object mesh, np.ndarray[dtype] point, np.ndarray[dtype] normal):
    # face points lie to left or right of plane
    cdef np.ndarray[int, ndim=2] faces = mesh.faces
    cdef np.ndarray[dtype, ndim=2] points = mesh.points
    cdef np.ndarray[int] owner = mesh.origMesh.owner
    cdef np.ndarray[int] neighbour = mesh.origMesh.neighbour
    cdef int nInternalCells = mesh.origMesh.nInternalCells
    cdef int nInternalFaces = mesh.origMesh.nInternalFaces
    cdef int d = faces.shape[1]-1

    left = (points[faces[:,1:]]-point).dot(normal) > 0.
    counter = left.sum(axis=1)
    inter = np.where((counter > 0) & (counter < d))[0]
    cdef int n = inter.shape[0]
    cdef np.ndarray[int, ndim=2] lines = -np.ones((n, 4), np.int32)

    cdef int i, j, k
    cdef int truth

    # get lines of intersection
    for i in range(0, n):
        j = inter[i]
        if (counter[j] == 1) or (counter[j] == (d-1)):
            truth = (counter[j] == 1)
            for k in range(0, d):
                if left[j,k] == truth:
                    break
            lines[i,0] = k
            lines[i,1] = (k-1)%d
            lines[i,2] = k
            lines[i,3] = (k+1)%d
        else:
            lines[i,0] = 0
            lines[i,2] = 2
            for k in [0, 2]:
                if left[j,(k-1)%d] != left[j,k]:
                    lines[i,k+1] = (k-1)%d
                else:
                    lines[i,k+1] = (k+1)%d

    # get points of intersection
    interPoints = np.zeros((n, 2, 3))
    for i in [0, 2]:
        l0 = points[faces[inter, 1 + lines[:,i]]]
        l1 = points[faces[inter, 1 + lines[:,i+1]]]
        l = (l1-l0)
        t = ((point-l0).dot(normal)/l.dot(normal)).reshape((-1, 1))
        interPoints[:,i/2,:] = l0 + t*l

    # get intersected cells
    internalInter = inter[inter < nInternalFaces]
    interCells = np.unique(np.concatenate((owner[inter], neighbour[internalInter])))
    interCellsMap = np.zeros(nInternalCells, np.int32)
    interCellsMap[interCells] = np.indices(interCells.shape, dtype=np.int32)
    interCellFaces = -np.ones((len(interCells), 4 + 1), np.int32)
    interCellFaces[:, 0] = 0
    # get intersected cell faces
    for i in range(0, len(inter)):
        cell = interCellsMap[owner[inter[i]]]
        curr = interCellFaces[cell,0]
        interCellFaces[cell,curr+1] = i
        interCellFaces[cell,0] += 1
        if inter[i] < nInternalFaces:
            cell = interCellsMap[neighbour[inter[i]]]
            curr = interCellFaces[cell,0]
            interCellFaces[cell,curr+1] = i
            interCellFaces[cell,0] += 1
            
    # get intersection area
    interCellPoints = interPoints[interCellFaces[:,1:]].reshape((-1, 2*4, 3))
    triangles = np.where(interCellFaces[:,0] == 3)[0]
    interCellPoints[triangles, -2:, :] = np.array([1e100,1e100,1e100])
    interDist1 = np.linalg.norm(interCellPoints-interCellPoints[:,[0],:], axis=-1)[:,1:].argmin(axis=1) + 1
    interDist2 = np.linalg.norm(interCellPoints-interCellPoints[:,[1],:], axis=-1)[:,2:].argmin(axis=1) + 2
    interDist1 += np.int32(((interDist1 % 2 == 0)-0.5)*2)
    interDist2 += np.int32(((interDist2 % 2 == 0)-0.5)*2)
    a = interCellPoints[np.arange(len(interCells)),interDist1,:]-interCellPoints[:,1,:]
    b = interCellPoints[np.arange(len(interCells)),interDist2,:]-interCellPoints[:,0,:]
    area = np.linalg.norm(np.cross(a, b), axis=1)/2
    return interCells, area.reshape((-1, 1))
        
            
@cython.boundscheck(False)
def getCells(object mesh):
    cdef np.ndarray[dtype, ndim=2] points = mesh.points
    cdef np.ndarray[int, ndim=2] cellFaces = mesh.cellFaces
    cdef np.ndarray[int, ndim=2] faces = mesh.faces
    cdef int nCells = cellFaces.shape[0]
    assert cellFaces.shape[1] == 6

    cdef np.ndarray[int, ndim=2] cellPoints = np.zeros((nCells, 8), dtype=np.int32)
    cdef int firstFace[4]
    cdef int nextFace[4]

    cdef int point, found
    cdef int i, j, k, l, m, n

    for i in range(0, nCells):
        for j in range(0, 4):
            firstFace[j] = faces[cellFaces[i, 0], 1+j]
        for j in range(0, 4):
            point = firstFace[j]
            found = 0
            for n in range(1, 6):
                for k in range(0, 4):
                    nextFace[k] = faces[cellFaces[i, n], 1+k]
                for k in range(0, 4):
                    if nextFace[k] == point:
                        l = (k + 1) % 4
                        for m in range(0, 4):
                            if firstFace[m] == nextFace[l]:
                                l = (k - 1) % 4
                                break
                        cellPoints[i,4+j] = nextFace[l]
                        found = 1
                        break
                if found:
                    break

        for j in range(0, 4):
            cellPoints[i,j] = firstFace[j]

    return cellPoints

