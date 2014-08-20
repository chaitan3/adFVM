from __future__ import print_function
import re
import numpy as np
import numpad as ad
import time

from numpad import adsparse
from scipy import sparse as sp

from field import Field
import utils
logger = utils.logger(__name__)

class Mesh(object):
    def __init__(self, caseDir):
        start = time.time()
        print('Reading mesh')

        self.case = caseDir
        meshDir = caseDir + '/constant/polyMesh/'
        self.faces = self.read(meshDir + 'faces', int)
        self.points = self.read(meshDir + 'points', float)
        self.owner = self.read(meshDir + 'owner', int).ravel()
        self.neighbour = self.read(meshDir + 'neighbour', int).ravel()
        self.boundary = self.readBoundary(meshDir + 'boundary')
        self.defaultBoundary = self.getDefaultBoundary()

        self.nInternalFaces = len(self.neighbour)
        self.nFaces = len(self.owner)
        self.nBoundaryFaces = self.nFaces-self.nInternalFaces
        self.nInternalCells = np.max(self.owner)+1
        self.nGhostCells = self.nBoundaryFaces
        self.nCells = self.nInternalCells + self.nGhostCells

        self.normals = self.getNormals()
        self.Normals = Field('nF', self, ad.array(self.normals))
        self.faceCentres, self.areas = self.getFaceCentresAndAreas()
        # uses neighbour
        self.cellFaces = self.getCellFaces()     # nInternalCells
        self.cellCentres, self.volumes = self.getCellCentresAndVolumes() # nCells after ghost cell mod
        # uses neighbour
        self.sumOp = self.getSumOp()             # (nInternalCells, nFaces)
        
        # ghost cell modification
        self.createGhostCells()
        self.deltas = self.getDeltas()           # nFaces
        self.weights = self.getWeights()   # nFaces

        end = time.time()
        print('Time for reading mesh:', end-start)
        print()

    def read(self, foamFile, dtype):
        logger.info('read {0}'.format(foamFile))
        lines = open(foamFile).readlines()
        first, last = 0, -1
        while lines[first][0] != '(': 
            first += 1
        while lines[last][0] != ')': 
            last -= 1
        f = lambda x: list(filter(None, re.split('[ ()\n]+', x)))
        return np.array(list(map(f, lines[first+1:last])), dtype)

    def readBoundary(self, boundaryFile):
        logger.info('read {0}'.format(boundaryFile))
        content = utils.removeCruft(open(boundaryFile).read())
        patches = re.findall(re.compile('([A-Za-z0-9_]+)[\r\s\n]+{(.*?)}', re.DOTALL), content)
        boundary = {}
        for patch in patches:
            boundary[patch[0]] = dict(re.findall('\n[ \t]+([a-zA-Z]+)[ ]+(.*?);', patch[1]))
            boundary[patch[0]]['nFaces'] = int(boundary[patch[0]]['nFaces'])
            boundary[patch[0]]['startFace'] = int(boundary[patch[0]]['startFace'])
        return boundary

    def getNormals(self):
        logger.info('generated normals')
        v1 = self.points[self.faces[:,1]]-self.points[self.faces[:,2]]
        v2 = self.points[self.faces[:,2]]-self.points[self.faces[:,3]]
        normals = np.cross(v1, v2)
        return normals / np.linalg.norm(normals, axis=1).reshape(-1,1)

    def getCellFaces(self):
        logger.info('generated cell faces')
        enum = lambda x: np.column_stack((np.indices(x.shape)[0], x)) 
        combined = np.concatenate((enum(self.owner), enum(self.neighbour)))
        cellFaces = combined[combined[:,1].argsort(), 0]
        # todo: make it a list ( investigate np.diff )
        return cellFaces.reshape(self.nInternalCells, len(cellFaces)/self.nInternalCells)

    def getCellCentresAndVolumes(self):
        logger.info('generated cell centres and volumes')
        nCellFaces = self.cellFaces.shape[1]
        cellCentres = np.mean(self.faceCentres[self.cellFaces], axis=1)
        sumCentres = cellCentres*0
        sumVolumes = np.sum(sumCentres, axis=1).reshape(-1,1)
        areaNormals = self.areas * self.normals
        for index in range(0, nCellFaces):
            indices = self.cellFaces[:,index]
            height = cellCentres-self.faceCentres[indices]
            volumes = np.abs(np.sum(areaNormals[indices]*height, axis=1)).reshape(-1,1)/3
            centres = (3./4)*self.faceCentres[indices] + (1./4)*cellCentres
            sumCentres += volumes * centres
            sumVolumes += volumes
        cellCentres = sumCentres/sumVolumes
        return cellCentres, sumVolumes

    def getFaceCentresAndAreas(self):
        logger.info('generated face centres and areas')
        nFacePoints = self.faces[0, 0]
        faceCentres = np.mean(self.points[self.faces[:,1:]], axis=1)
        sumAreas = 0
        sumCentres = 0
        for index in range(1, nFacePoints+1):
            points = self.points[self.faces[:, index]]
            nextPoints = self.points[self.faces[:, (index % nFacePoints)+1]]
            centres = (points + nextPoints + faceCentres)/3
            normals = np.cross((nextPoints - points), (faceCentres - points))
            areas = (np.linalg.norm(normals, axis=1)/2).reshape(-1,1)
            sumAreas += areas
            sumCentres += areas*centres
        faceCentres = sumCentres/sumAreas
        return faceCentres, sumAreas

    def getDeltas(self):
        logger.info('generated deltas')
        P = self.cellCentres[self.owner]
        N = self.cellCentres[self.neighbour]
        return np.linalg.norm(P-N, axis=1).reshape(-1,1)

    def getWeights(self):
        logger.info('generated face deltas')
        P = self.cellCentres[self.owner]
        N = self.cellCentres[self.neighbour]
        F = self.faceCentres
        neighbourDist = np.abs(np.sum((F-N)*self.normals, axis=1))
        ownerDist = np.abs(np.sum((F-P)*self.normals, axis=1))
        weights = neighbourDist/(neighbourDist + ownerDist)
        return weights.reshape(-1,1)

    def getSumOp(self):
        logger.info('generated sum op')
        owner = sp.csc_matrix((np.ones(self.nFaces), self.owner, range(0, self.nFaces+1)), shape=(self.nInternalCells, self.nFaces))
        Nindptr = np.concatenate((range(0, self.nInternalFaces+1), self.nInternalFaces*np.ones(self.nFaces-self.nInternalFaces, int)))
        neighbour = sp.csc_matrix((-np.ones(self.nInternalFaces), self.neighbour[:self.nInternalFaces], Nindptr), shape=(self.nInternalCells, self.nFaces))
        sumOp = (owner + neighbour).tocsr()
        return adsparse.csr_matrix((ad.array(sumOp.data), sumOp.indices, sumOp.indptr), sumOp.shape)

    def getDefaultBoundary(self):
        logger.info('generated default boundary')
        boundary = {}
        for patchID in self.boundary:
            boundary[patchID] = {}
            if self.boundary[patchID]['type'] in ['cyclic', 'symmetryPlane', 'empty']:
                boundary[patchID]['type'] = self.boundary[patchID]['type']
            else:
                boundary[patchID]['type'] = 'zeroGradient'
        return boundary

    def createGhostCells(self):
        logger.info('generated ghost cells')
        self.neighbour = np.concatenate((self.neighbour, np.zeros(self.nBoundaryFaces, int)))
        self.cellCentres = np.concatenate((self.cellCentres, np.zeros((self.nBoundaryFaces, 3))))
        for patchID in self.boundary:
            patch = self.boundary[patchID]
            startFace = patch['startFace']
            nFaces = patch['nFaces']
            endFace = startFace + nFaces
            indices = self.nInternalCells + range(startFace, endFace) - self.nInternalFaces 
            # append neighbour
            self.neighbour[startFace:endFace] = indices
            if patch['type'] == 'cyclic': 
                neighbourPatch = self.boundary[patch['neighbourPatch']]   
                neighbourStartFace = neighbourPatch['startFace']
                neighbourEndFace = neighbourStartFace + nFaces
                # apply transformation
                # append cell centres
                patch['transform'] = self.faceCentres[startFace]-self.faceCentres[neighbourStartFace]
                self.cellCentres[indices] = patch['transform'] + self.cellCentres[self.owner[neighbourStartFace:neighbourEndFace]]
            else:
                # append cell centres
                self.cellCentres[indices] = self.faceCentres[startFace:endFace]
             
