cimport numpy as np
cimport cython
from cython.parallel import prange, parallel
#cimport part_mesh
from libcpp.vector cimport vector
from libcpp.set cimport set
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.string cimport string
from libc.stdio cimport sprintf

import copy
import numpy as np
import scipy as sp
ctypedef fused dtype:
    cython.double
    cython.float


@cython.boundscheck(False)
def intersectPlane(object mesh, np.ndarray[dtype] point, np.ndarray[dtype] normal):
    # face points lie to left or right of plane
    cdef np.ndarray[int, ndim=2] faces = mesh.faces
    cdef np.ndarray[dtype, ndim=2] points = mesh.points
    cdef np.ndarray[int] owner = mesh.origMesh.owner
    cdef np.ndarray[int] neighbour = mesh.origMesh.neighbour
    cdef int nInternalCells = mesh.nInternalCells
    cdef int nInternalFaces = mesh.nInternalFaces
    cdef int nFaces = mesh.nFaces
    cdef int d = faces.shape[1]-1
    cdef int i, j, k

    import time
    start = time.time()
    cdef np.ndarray[dtype, ndim=2] left = np.zeros((nFaces, d), point.dtype)
    cdef np.ndarray[int] counter = np.zeros(nFaces, np.int32)
    with nogil, parallel():
        for i in prange(0, nFaces):
            for j in range(1, d+1):
                left[i,j-1] = 0.
                for k in range(0, 3):
                    left[i,j-1] += (points[faces[i,j], k]-point[k])*normal[k]
                counter[i] += left[i,j-1] > 0.


    #left = (points[faces[:,1:]]-point).dot(normal) > 0.
    #counter = left.sum(axis=1)
    inter = np.where((counter > 0) & (counter < d))[0]
    print(time.time()-start)
    cdef int n = inter.shape[0]
    cdef np.ndarray[int, ndim=2] lines = -np.ones((n, 4), np.int32)

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
    print(time.time()-start)

    # get points of intersection
    interPoints = np.zeros((n, 2, 3), mesh.points.dtype)
    for i in [0, 2]:
        l0 = points[faces[inter, 1 + lines[:,i]]]
        l1 = points[faces[inter, 1 + lines[:,i+1]]]
        l = (l1-l0)
        t = ((point-l0).dot(normal)/l.dot(normal)).reshape((-1, 1))
        interPoints[:,i/2,:] = l0 + t*l
    print(time.time()-start)

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
    print(time.time()-start)
            
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
    print(time.time()-start)
    return interCells, area.reshape((-1, 1))
        
            
@cython.boundscheck(False)
def getCells(object mesh, np.ndarray[dtype] infer):
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

cdef string int_string(int val):
    cdef char buff[6]
    sprintf(buff, "%d", val)
    cdef string s = string(buff)
    return s

cdef vector_to_numpy(vector[int] &arr):
    cdef int el, i = 0
    cdef np.ndarray[int, ndim=1] np_arr = np.zeros(arr.size(), np.int32)
    for el in arr:
        np_arr[i] = el
        i += 1
    return np_arr

cdef set_to_numpy(set[int] &arr):
    cdef int el, i = 0
    cdef np.ndarray[int] np_arr = np.zeros(arr.size(), np.int32)
    for el in arr:
        np_arr[i] = el
        i += 1
    return np_arr

@cython.boundscheck(False)
def decompose(object mesh, int nprocs, np.ndarray[dtype] infer):

    meshO = mesh.origMesh
    ne = meshO.nInternalCells
    nn = mesh.points.shape[0]
    cdef np.ndarray[int, ndim=1] eptr = np.arange(0, ne*8+1, 8, dtype=np.int32)
    cdef np.ndarray[int, ndim=2] eind = mesh.cells.astype(np.int32)

    cdef np.ndarray[int, ndim=1] epart = np.zeros(ne, np.int32)
    cdef np.ndarray[int, ndim=1] npart = np.zeros(nn, np.int32)
    cdef int ncommon = 4
    cdef int c_ne = ne
    cdef int c_nn = nn
    cdef int objval
    cdef int[40] options
    #part_mesh.METIS_SetDefaultOptions(options)
    #options[0] = 1
    #options[1] = 0
    #options[2] = 1
    #options[3] = 4
    #options[4] = 1
    #options[16] = 30
    #part_mesh.METIS_PartMeshDual(&c_ne, &c_nn, &eptr[0], &eind[0,0], NULL, NULL, &ncommon, &nprocs, NULL, options, &objval, &epart[0], &npart[0])

    print 'metis completed'

    cdef np.ndarray[int, ndim=1] owner = meshO.owner
    cdef np.ndarray[int, ndim=1] neighbour = meshO.neighbour
    cdef np.ndarray[int, ndim=2] cells = mesh.cells
    cdef np.ndarray[dtype, ndim=2] points = mesh.points
    cdef np.ndarray[int, ndim=2] faces = mesh.faces
    boundary = meshO.boundary

    cdef int nFaces = meshO.nFaces
    cdef int nInternalFaces = meshO.nInternalFaces
    cdef int nInternalCells = meshO.nInternalCells
    cdef int i, j, k, l, m, n, o

    cdef vector[vector[int]] faceProc
    cdef vector[vector[int]] cellProc
    cdef vector[set[int]] pointProc
    cdef vector[map[string, vector[int]]] boundaryProc
    cdef vector[map[string, map[string, string]]] remoteBoundaryProc
    cdef vector[vector[string]] boundaryProcOrder

    cdef vector[int] temp
    cdef set[int] temp2
    cdef map[string, vector[int]] temp3
    cdef map[string, map[string, string]] temp4
    cdef map[string, string] temp5
    cdef vector[string] temp6

    cdef pair[string, string] p_kv
    cdef pair[string, vector[int]] p_patch
    cdef pair[string, map[string, string]] p_remote
    cdef string patch, newPatch
    cdef string* buff = [string("type"),
                            string("processor"),
                            string("myProcNo"),
                            string("neighbProcNo"),
                            string("processorCyclic"),
                            string("referPatch")]

    cdef char[100] c_patch
    cdef vector[int] internalFaceProc

    for i in range(0, nprocs):
        faceProc.push_back(temp)
        cellProc.push_back(temp)
        pointProc.push_back(temp2)
        boundaryProc.push_back(temp3)
        remoteBoundaryProc.push_back(temp4)
        boundaryProcOrder.push_back(temp6)

    for i in range(0, nInternalCells):
        cellProc[epart[i]].push_back(i)

    for patchID in meshO.boundary:
        meshPatch = meshO.boundary[patchID]
        patch = patchID
        k = meshPatch['startFace']
        l = k + meshPatch['nFaces']
        n = 0
        if meshPatch['type'] == 'cyclic':
            neighbourPatch = meshO.boundary[meshPatch['neighbourPatch']]
            m = neighbourPatch['startFace']
            n = 1
        for i in range(0, nprocs):
            boundaryProc[i][patch] = temp
            boundaryProcOrder[i].push_back(patch)
        for i in range(k, l):
            j = epart[owner[i]]
            if n:
                n = m + i - k
                o = epart[owner[n]]
                if j == o:
                    boundaryProc[j][patch].push_back(i)
                else:
                    sprintf(c_patch, "procBoundary%dto%dthrough%s", j, o, patch.c_str())
                    newPatch.assign(c_patch)
                    if boundaryProc[j].count(newPatch) == 0:
                        boundaryProc[j][newPatch] = temp
                        remoteBoundaryProc[j][newPatch] = temp5
                        remoteBoundaryProc[j][newPatch][buff[0]] = buff[4]
                        remoteBoundaryProc[j][newPatch][buff[2]] = int_string(j)
                        remoteBoundaryProc[j][newPatch][buff[3]] = int_string(o)
                        remoteBoundaryProc[j][newPatch][buff[5]] = patch
                    boundaryProc[j][newPatch].push_back(i)
            else:
                boundaryProc[j][patch].push_back(i)

    for i in range(0, nInternalFaces):
        j = epart[owner[i]]  
        k = epart[neighbour[i]]  
        if j == k:
            faceProc[j].push_back(i)
        else:
            sprintf(c_patch, "procBoundary%dto%d", j, k)
            patch.assign(c_patch)
            if boundaryProc[j].count(patch) == 0:
                boundaryProc[j][patch] = temp
                remoteBoundaryProc[j][patch] = temp5
                remoteBoundaryProc[j][patch][buff[0]] = buff[1]
                remoteBoundaryProc[j][patch][buff[2]] = int_string(j)
                remoteBoundaryProc[j][patch][buff[3]] = int_string(k)
            boundaryProc[j][patch].push_back(i)

            sprintf(c_patch, "procBoundary%dto%d", k, j)
            patch.assign(c_patch)
            if boundaryProc[k].count(patch) == 0:
                boundaryProc[k][patch] = temp
                remoteBoundaryProc[k][patch] = temp5
                remoteBoundaryProc[k][patch][buff[0]] = buff[1]
                remoteBoundaryProc[k][patch][buff[2]] = int_string(k)
                remoteBoundaryProc[k][patch][buff[3]] = int_string(j)
            boundaryProc[k][patch].push_back(i)

    for i in range(0, nprocs):
        for p_remote in remoteBoundaryProc[i]:
            patch = p_remote.first
            boundaryProcOrder[i].push_back(patch)

    for i in range(0, nprocs):
        internalFaceProc.push_back(faceProc[i].size())
        for patch in boundaryProcOrder[i]:
            for j in range(0, boundaryProc[i][patch].size()):
                faceProc[i].push_back(boundaryProc[i][patch][j])

    for i in range(0, nprocs):
        temp = cellProc[i]
        for j in range(0, temp.size()):
            for k in range(0, 8):
                pointProc[i].insert(cells[temp[j],k])

    print 'mappings completed'

    decomposed = []
    addressing = []
    cdef np.ndarray[dtype, ndim=2] procPoints
    cdef np.ndarray[int, ndim=2] procFaces
    cdef np.ndarray[int, ndim=1] procOwner
    cdef np.ndarray[int, ndim=1] procNeighbour
    cdef map[int, int] revPointProc
    cdef map[int, int] revCellProc
    for i in range(0, nprocs):
        revPointProc.clear()
        revCellProc.clear()
        procBoundary = copy.deepcopy(boundary)

        j = pointProc[i].size()
        procPoints = np.zeros((j, 3), np.float64)
        k = 0
        for it in pointProc[i]:
            for l in range(0, 3):
                procPoints[k, l] = points[it, l]
            revPointProc[it] = k
            k += 1

        k = 0
        for it in cellProc[i]:
            revCellProc[it] = k
            k += 1 

        j = faceProc[i].size()
        procFaces = np.zeros((j, 5), np.int32)
        procFaces[:,0] = 4
        procOwner = np.zeros(j, np.int32)
        procNeighbour = np.zeros(internalFaceProc[i], np.int32)
        k = 0
        for it in faceProc[i]:
            if epart[owner[it]] == i:
                procOwner[k] = revCellProc[owner[it]]
                for l in range(0, 4):
                    procFaces[k, l+1] = revPointProc[faces[it, l+1]]
            else:
                procOwner[k] = revCellProc[neighbour[it]]
                for l in range(0, 4):
                    procFaces[k, l+1] = revPointProc[faces[it, 3-l+1]]
            if k < internalFaceProc[i]:
                procNeighbour[k] = revCellProc[neighbour[it]]
            k += 1

        j = internalFaceProc[i]
        for patch in boundaryProcOrder[i]:
            k = boundaryProc[i][patch].size()
            if str(patch) not in procBoundary:
                procBoundary[patch] = {}
            procBoundary[patch].pop("neighbourIndices", None)
            procBoundary[patch]['startFace'] = j
            procBoundary[patch]['nFaces'] = k
            j += k

        for p_remote in remoteBoundaryProc[i]:
            patch = p_remote.first
            for p_kv in p_remote.second:
                procBoundary[patch][p_kv.first] = p_kv.second

        decomposed.append((procPoints, procFaces, procOwner, procNeighbour, procBoundary))
        # convert to ndarray
        pyPointProc = set_to_numpy(pointProc[i])
        pyFaceProc = vector_to_numpy(faceProc[i])
        pyCellProc = vector_to_numpy(cellProc[i])
        pyBoundaryProc = []
        for patch in boundaryProcOrder[i]:
            try:
                order = mesh.localPatches.index(patch)
            except ValueError:
                order = -1
            pyBoundaryProc.append(order)
        pyBoundaryProc = np.array(pyBoundaryProc, np.int32)

        addressing.append((pyPointProc, pyFaceProc, pyCellProc, pyBoundaryProc))
    return decomposed, addressing

def reduceAbsMin(np.ndarray[dtype, ndim=2] arr, np.ndarray[int] count):
    cdef int n = count.shape[0]
    cdef int m = arr.shape[1]
    cdef np.ndarray[dtype, ndim=2] arrMin = np.zeros((n, m), arr.dtype)
    cdef np.ndarray[int, ndim=2] arrArgMin = np.zeros((n, m), count.dtype)
    cdef int i, j, k, l
    cdef int c, d
    cdef dtype absMin
    d = 0
    for i in range(0, n):
        c = count[i]
        for j in range(0, m):
            absMin = 1e20
            for k in range(0, c):
                if abs(arr[d+k,j]) < absMin:
                    l = k
                    absMin = abs(arr[d+k,j])
            arrMin[i,j] = arr[d+l,j]
            arrArgMin[i,j] = d+l
        d += c
    return arrMin, arrArgMin

def reduceSum(np.ndarray[dtype, ndim=2] arr, np.ndarray[int] count):
    cdef int n = count.shape[0]
    cdef int m = arr.shape[1]
    cdef np.ndarray[dtype, ndim=2] arrSum = np.zeros((n, m), arr.dtype)
    cdef int i, j, k, l
    cdef int c, d
    d = 0
    for i in range(0, n):
        c = count[i]
        for j in range(0, m):
            absMin = 1e20
            for k in range(0, c):
                arrSum[i, j] += arr[d+k,j]
        d += c
    return arrSum

def selectMultipleRange(np.ndarray[int] start, np.ndarray[int] count):
    cdef int total = count.sum()
    cdef np.ndarray[int] arr = np.zeros(total, np.int32)
    cdef int n = count.shape[0]
    cdef int i, j
    cdef int pos = 0
    for i in range(0, n):
        for j in range(0, count[i]):
            arr[pos] = start[i]+j
            pos += 1
    return arr

def ARLogLikelihood(np.ndarray[dtype] params, np.ndarray[dtype, ndim=2] s, int p):

    cdef int n = len(s) + 1
    cdef int n2 = n*n
    cdef int n3 = n*n*n
    cdef int Np = n*p + (n-1)*p + n
    cdef int N = len(s[0])

    cdef np.ndarray[dtype, ndim=2] a = params[:n*p].reshape(p, n)
    cdef np.ndarray[dtype, ndim=2] b = params[n*p:(2*n-1)*p].reshape(p, n-1)
    cdef np.ndarray[dtype] d = params[(2*n-1)*p:(2*n-1)*p + n]

    cdef np.ndarray[dtype, ndim=3] A = np.zeros((p, n, n))
    cdef np.ndarray[dtype, ndim=2] B = np.zeros(((p+1)*n*n, (p+1)*n*n))
    ii = np.arange(0, n)
    A[:,ii,ii] = a
    A[:,:n-1,-1] = b

    E = np.diag(d)
    #cdef np.ndarray[dtype, ndim=2] E = np.diag(d)

    cdef int i, j, k, l, m
    cdef int index
    print a, b, d
    for i in range(0, p+1):
        for j in range(i*n2, (i+1)*n2):
            B[j, j] += 1
        index = 0
        for j in range(i-1, -1, -1):
            for m in range(0, n):
                for k in range(0, n):
                    for l in range(0, n):
                        B[i*n2 + m*n + k, j*n2 + m*n + l] += -A[index, k, l]
            index += 1
        for j in range(1, p-i+1):
            for m in range(0, n):
                for k in range(0, n):
                    for l in range(0, n):
                        B[i*n2 + m*n + k, j*n2 + l*n + m] += -A[index, k, l]
            index += 1
    r = np.zeros((p+1)*n2)
    r[:n2] = E.flatten()
    Ej = np.linalg.solve(B, r)
    Ej = Ej.reshape(p+1, n2)
    cdef np.ndarray[dtype, ndim=2] Es = np.zeros((N, n2))
    #print Ej
    Es[:p+1] = Ej
    for i in range(p+1, N):
        for j in range(i-1, i-p-1, -1):
            for k in range(0, n):
                for l in range(0, n):
                    for m in range(0, n):
                        Es[i, l*n + k] += A[i-j-1, k, m]*Es[j, l*n + m]
        #print np.abs(Es).max()
    cdef int n1 = n-1
    cdef np.ndarray[dtype, ndim=2] S = np.zeros((N*n1, N*n1))
    #print (np.abs(Es)).max(axis=1)
    for i in range(0, N):
        for j in range(i, N):
            for k in range(0, n1):
                for l in range(0, n1):
                    S[i*n1 + k, j*n1 + l] = Es[j-i, l*n + k]
    S = np.triu(S) + np.triu(S, k=1).T
    R = sp.linalg.cholesky(S)
    Y = (s.T)[::-1].flatten()
    z1 = np.dot(Y, sp.linalg.cho_solve((R, False), Y))
    z2 = 2*np.log(np.diag(R)).sum()
    print z1, z2
    res = 0.5*z1 + 0.5*z2
    return res

