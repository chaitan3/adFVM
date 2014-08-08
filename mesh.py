import re
import numpy as np
from scipy import sparse as sp

class Mesh:
    def __init__(self, caseDir):
        meshDir = caseDir + '/constant/polyMesh/'
        self.faces = self.read(meshDir + 'faces', int)
        self.points = self.read(meshDir + 'points', float)
        self.owner = self.read(meshDir + 'owner', int).ravel()
        self.neighbour = self.read(meshDir + 'neighbour', int).ravel()

        self.nInternalFaces = len(self.neighbour)
        self.nFaces = len(self.owner)
        self.nCells = np.max(self.owner)+1

        self.normals = self.getNormals()
        self.areas = self.getAreas()
        self.faceCentres = self.getFaceCentres()
        self.cellFaces = self.getCellFaces()
        self.cellCentres = self.getCellCentres()
        self.volumes = self.getVolumes()
        self.deltas = self.getDeltas()
        self.faceDeltas = self.getFaceDeltas()
        self.sumOp = self.getSumOp()

    def read(self, foamFile, dtype):
        print 'read', foamFile
        lines = open(foamFile).readlines()
        first, last = 0, -1
        while lines[first][0] != '(': 
            first += 1
        while lines[last][0] != ')': 
            last -= 1
        f = lambda x: filter(None, re.split('[ ()\n]+', x))
        return np.array(map(f, lines[first+1:last]), dtype)

    def getNormals(self):
        print 'generated normals'
        #correct direcion? mostly yes
        v1 = self.points[self.faces[:,1]]-self.points[self.faces[:,2]]
        v2 = self.points[self.faces[:,2]]-self.points[self.faces[:,3]]
        normals = np.cross(v1, v2)
        return normals / np.linalg.norm(normals, axis=1).reshape(-1,1)

    def getAreas(self):
        print 'generated areas'
        nFacePoints = self.faces[0, 0]
        areas = np.cross(self.points[self.faces[:,-1]], self.points[self.faces[:,1]])
        for i in range(0, nFacePoints-1):
            areas += np.cross(self.points[self.faces[:,i+1]], self.points[self.faces[:,i+2]])
        return np.linalg.norm(areas, axis=1)/2

    def getVolumes(self):
        print 'generated volumes'
        nCellFaces = self.cellFaces.shape[1]
        volumes = 0
        for i in range(0, nCellFaces):
            legs = self.points[self.faces[self.cellFaces[:,i], 1:]]-self.cellCentres.reshape((self.nCells, 1, 3))
            volumes += np.abs(np.sum(np.cross(legs[:,0,:], legs[:,2,:])*(legs[:,1,:]-legs[:,3,:]), axis=1))/6
        return volumes

    def getCellFaces(self):
        print 'generated cell faces'
        #slow
        cellFaces = np.zeros((self.nCells, 6), int)
        for i in range(0, self.nCells):
            cellFaces[i] = np.concatenate((np.where(self.owner == i)[0], np.where(self.neighbour == i)[0]))
        return cellFaces

    def getCellCentres(self):
        print 'generated cell centres'
        return np.mean(self.faceCentres[self.cellFaces], axis=1)

    def getFaceCentres(self):
        print 'generated face centres'
        return np.mean(self.points[self.faces[:,1:]], axis=1)

    def getDeltas(self):
        print 'generated deltas'
        P = self.cellCentres[self.owner[:self.nInternalFaces]]
        N = self.cellCentres[self.neighbour]
        return np.linalg.norm(P-N, axis=1)

    def getFaceDeltas(self):
        print 'generated face deltas'
        P = self.faceCentres[:self.nInternalFaces]
        N = self.cellCentres[self.neighbour]
        return np.linalg.norm(P-N, axis=1)

    def getSumOp(self):
        print 'generated sum op'
        owner = sp.csc_matrix((np.ones(self.nFaces), self.owner, range(0, self.nFaces+1)), shape=(self.nCells, self.nFaces))
        Nindptr = np.concatenate((range(0, self.nInternalFaces+1), self.nInternalFaces*np.ones(self.nFaces-self.nInternalFaces, int)))
        neighbour = sp.csc_matrix((-np.ones(self.nInternalFaces), self.neighbour, Nindptr), shape=(self.nCells, self.nFaces))
        return (owner + neighbour).tocsr()

             
