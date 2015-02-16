import numpy as np
from scipy import sparse as sp
import re
import time
import copy

from config import ad, adsparse, T, Logger
from parallel import pprint, Exchanger
logger = Logger(__name__)
import config, parallel

class Mesh(object):
    def __init__(self, caseDir=None):
        if caseDir is None:
            self.owner = self.neighbour = None
            self.sumOp = None
            self.areas = self.volumes = self.weights = self.normals = None
            self.localRemoteCells = None
            self.localRemoteFaces = None
            self.remoteCells = None
            return

        start = time.time()
        pprint('Reading mesh')

        self.case = caseDir + parallel.processorDirectory
        meshDir = self.case + 'constant/polyMesh/'
        self.faces = self.read(meshDir + 'faces', np.int32)
        self.points = self.read(meshDir + 'points', np.float64).astype(config.precision)
        self.owner = self.read(meshDir + 'owner', np.int32).ravel()
        self.neighbour = self.read(meshDir + 'neighbour', np.int32).ravel()
        self.boundary, self.localPatches, self.remotePatches = self.readBoundary(meshDir + 'boundary')
        self.origPatches = copy.copy(self.localPatches)
        self.origPatches.sort()
        self.defaultBoundary = self.getDefaultBoundary()
        self.calculatedBoundary = self.getCalculatedBoundary()

        self.nInternalFaces = len(self.neighbour)
        self.nFaces = len(self.owner)
        self.nBoundaryFaces = self.nFaces-self.nInternalFaces
        self.nInternalCells = np.max(self.owner)+1
        self.nGhostCells = self.nBoundaryFaces
        self.nCells = self.nInternalCells + self.nGhostCells

        self.normals = self.getNormals()
        self.faceCentres, self.areas = self.getFaceCentresAndAreas()
        # uses neighbour
        self.cellFaces = self.getCellFaces()     # nInternalCells
        self.cellCentres, self.volumes = self.getCellCentresAndVolumes() # nCells after ghost cell mod
        # uses neighbour
        self.sumOp = self.getSumOp(self)             # (nInternalCells, nFaces)
        self.absSumOp = self.getAbsSumOp()             # (nInternalCells, nFaces)

        
        # ghost cell modification
        self.nLocalCells = self.createGhostCells()
        self.deltas = self.getDeltas()           # nFaces
        self.weights = self.getWeights()   # nFaces

        # padded mesh
        self.paddedMesh = self.createPaddedMesh()

        # theano shared variables
        #self.owner = T.shared(self.owner)
        #self.neighbour = T.shared(self.neighbour)
        #self.normals = T.shared(self.normals)
        #self.areas = T.shared(self.areas)
        #self.volumes = T.shared(self.volumes)
        #self.cellCentres = T.shared(self.cellCentres)
        #self.deltas = T.shared(self.deltas)
        #self.weights = T.shared(self.weights)

        end = time.time()
        pprint('Time for reading mesh:', end-start)
        pprint()

    def read(self, foamFile, dtype):
        logger.info('read {0}'.format(foamFile))
        content = open(foamFile).read()
        foamFileDict = re.search(re.compile('FoamFile\n{(.*?)}\n', re.DOTALL), content).group(1)
        assert re.search('format[\s\t]+(.*?);', foamFileDict).group(1) == config.fileFormat
        start = content.find('(') + 1
        end = content.rfind(')')
        if config.fileFormat == 'binary':
            if foamFile[-5:] == 'faces':
                nFaces1 = int(re.search('[0-9]+', content[start-2:0:-1]).group(0)[::-1])
                endIndices = start + nFaces1*4
                faceIndices = np.fromstring(content[start:endIndices], dtype)
                faceIndices = faceIndices[1:] - faceIndices[:-1]
                startData = content.find('(', endIndices) + 1
                data = np.fromstring(content[startData:end], dtype)
                nCellFaces = faceIndices[0] 
                return np.hstack((faceIndices.reshape(-1, 1), data.reshape(len(data)/nCellFaces, nCellFaces)))
            else:
                data = np.fromstring(content[start:end], dtype)
                if foamFile[-6:] == 'points':
                    data = data.reshape(len(data)/3, 3)
                return data
        else:
            f = lambda x: list(filter(None, re.split('[ ()\n]+', x)))
            return np.array(list(map(f, filter(None, re.split('\n', content[start:end])))), dtype)

    def readBoundary(self, boundaryFile):
        logger.info('read {0}'.format(boundaryFile))
        content = removeCruft(open(boundaryFile).read())
        patches = re.findall(re.compile('([A-Za-z0-9_]+)[\r\s\n]+{(.*?)}', re.DOTALL), content)
        boundary = {}
        localPatches = []
        remotePatches = []
        for patch in patches:
            boundary[patch[0]] = dict(re.findall('\n[ \t]+([a-zA-Z]+)[ ]+(.*?);', patch[1]))
            boundary[patch[0]]['nFaces'] = int(boundary[patch[0]]['nFaces'])
            boundary[patch[0]]['startFace'] = int(boundary[patch[0]]['startFace'])
            if boundary[patch[0]]['type'] in config.processorPatches:
                remotePatches.append(patch[0])
            else:
                localPatches.append(patch[0])
        return boundary, localPatches, remotePatches

    def getNormals(self):
        logger.info('generated normals')
        v1 = self.points[self.faces[:,1]]-self.points[self.faces[:,2]]
        v2 = self.points[self.faces[:,2]]-self.points[self.faces[:,3]]
        # CROSS product makes it F_CONTIGUOUS even if normals is not
        normals = np.cross(v1, v2)
        # change back to contiguous
        normals = np.ascontiguousarray(normals)
        return normals / config.norm(normals, axis=1).reshape(-1,1)

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
            areas = (config.norm(normals, axis=1)/2).reshape(-1,1)
            sumAreas += areas
            sumCentres += areas*centres
        faceCentres = sumCentres/sumAreas
        return faceCentres, sumAreas

    def getDeltas(self):
        logger.info('generated deltas')
        P = self.cellCentres[self.owner]
        N = self.cellCentres[self.neighbour]
        return config.norm(P-N, axis=1).reshape(-1,1)

    def getWeights(self):
        logger.info('generated face deltas')
        P = self.cellCentres[self.owner]
        N = self.cellCentres[self.neighbour]
        F = self.faceCentres
        neighbourDist = np.abs(np.sum((F-N)*self.normals, axis=1))
        ownerDist = np.abs(np.sum((F-P)*self.normals, axis=1))
        weights = neighbourDist/(neighbourDist + ownerDist)
        return weights.reshape(-1,1)

    def getSumOp(self, mesh):
        logger.info('generated sum op')
        owner = sp.csc_matrix((np.ones(mesh.nFaces), mesh.owner, range(0, mesh.nFaces+1)), shape=(mesh.nInternalCells, mesh.nFaces))
        Nindptr = np.concatenate((range(0, mesh.nInternalFaces+1), mesh.nInternalFaces*np.ones(mesh.nFaces-mesh.nInternalFaces, int)))
        neighbour = sp.csc_matrix((-np.ones(mesh.nInternalFaces), mesh.neighbour[:mesh.nInternalFaces], Nindptr), shape=(mesh.nInternalCells, mesh.nFaces))
        # skip empty patches
        for patchID in self.boundary:
            patch = self.boundary[patchID]
            if patch['type'] == 'empty' and patch['nFaces'] != 0:
                pprint('Deleting empty patch ', patchID)
                startFace = mesh.nInternalFaces + patch['startFace'] - self.nInternalFaces
                endFace = startFace + patch['nFaces']
                owner.data[startFace:endFace] = 0
        sumOp = (owner + neighbour).tocsr()
        #return sumOp
        return adsparse.CSR(sumOp.data, sumOp.indices, sumOp.indptr, sumOp.shape)

    def getAbsSumOp(self):
        logger.info('generated abs sum op')
        owner = sp.csc_matrix((np.ones(self.nFaces), self.owner, range(0, self.nFaces+1)), shape=(self.nInternalCells, self.nFaces))
        Nindptr = np.concatenate((range(0, self.nInternalFaces+1), self.nInternalFaces*np.ones(self.nFaces-self.nInternalFaces, int)))
        neighbour = sp.csc_matrix((np.ones(self.nInternalFaces), self.neighbour[:self.nInternalFaces], Nindptr), shape=(self.nInternalCells, self.nFaces))
        sumOp = (owner + neighbour).tocsr()
        #return sumOp
        return adsparse.CSR(sumOp.data, sumOp.indices, sumOp.indptr, sumOp.shape)

    def getDefaultBoundary(self):
        logger.info('generated default boundary')
        boundary = {}
        for patchID in self.boundary:
            boundary[patchID] = {}
            if self.boundary[patchID]['type'] in config.defaultPatches:
                boundary[patchID]['type'] = self.boundary[patchID]['type']
            else:
                boundary[patchID]['type'] = 'zeroGradient'
        return boundary

    def getCalculatedBoundary(self):
        logger.info('generated calculated boundary')
        boundary = {}
        for patchID in self.boundary:
            boundary[patchID] = {}
            if self.boundary[patchID]['type'] in config.coupledPatches:
                boundary[patchID]['type'] = self.boundary[patchID]['type']
            else:
                boundary[patchID]['type'] = 'calculated'
        return boundary

    def getProcessorPatchInfo(self, patchID):
        patch = self.boundary[patchID]
        local = patch['myProcNo']
        remote = patch['neighbProcNo']
        tag = 0
        if patch['type'] == 'processorCyclic':
            commonPatch = patch['referPatch']
            if local > remote:
                commonPatch = self.boundary[commonPatch]['neighbourPatch']
            tag = 1 + self.origPatches.index(commonPatch)
        return local, remote, tag



    def createGhostCells(self):
        logger.info('generated ghost cells')
        self.neighbour = np.concatenate((self.neighbour, np.zeros(self.nBoundaryFaces, np.int32)))
        self.cellCentres = np.concatenate((self.cellCentres, np.zeros((self.nBoundaryFaces, 3))))
        nLocalCells = self.nInternalCells
        exchanger = Exchanger()
        for patchID in self.boundary:
            patch = self.boundary[patchID]
            startFace = patch['startFace']
            nFaces = patch['nFaces']
            # empty patches
            if nFaces == 0:
                continue
            elif patch['type'] not in config.processorPatches:
                nLocalCells += nFaces
            endFace = startFace + nFaces
            cellStartFace = self.nInternalCells + startFace - self.nInternalFaces
            cellEndFace = self.nInternalCells + endFace - self.nInternalFaces
            # append neighbour
            self.neighbour[startFace:endFace] = range(cellStartFace, cellEndFace)
            if patch['type'] == 'cyclic': 
                neighbourPatch = self.boundary[patch['neighbourPatch']]   
                neighbourStartFace = neighbourPatch['startFace']
                neighbourEndFace = neighbourStartFace + nFaces
                # apply transformation: single value
                # append cell centres
                patch['transform'] = self.faceCentres[startFace]-self.faceCentres[neighbourStartFace]
                self.cellCentres[cellStartFace:cellEndFace] = patch['transform'] + self.cellCentres[self.owner[neighbourStartFace:neighbourEndFace]]

            elif patch['type'] == 'processor':
                patch['neighbProcNo'] = int(patch['neighbProcNo'])
                patch['myProcNo'] = int(patch['myProcNo'])
                local, remote, tag = self.getProcessorPatchInfo(patchID)
                # exchange data
                exchanger.exchange(remote, self.cellCentres[self.owner[startFace:endFace]], self.cellCentres[cellStartFace:cellEndFace], tag)

            elif patch['type'] == 'processorCyclic':
                patch['neighbProcNo'] = int(patch['neighbProcNo'])
                patch['myProcNo'] = int(patch['myProcNo'])
                local, remote, tag = self.getProcessorPatchInfo(patchID)
                # apply transformation
                exchanger.exchange(remote, -self.faceCentres[startFace:endFace] + self.cellCentres[self.owner[startFace:endFace]], self.cellCentres[cellStartFace:cellEndFace], tag)
            else:
                # append cell centres
                self.cellCentres[cellStartFace:cellEndFace] = self.faceCentres[startFace:endFace]
        exchanger.wait()
        for patchID in self.boundary:
            patch = self.boundary[patchID]
            startFace = patch['startFace']
            nFaces = patch['nFaces']
            # empty patches
            if nFaces == 0:
                continue
            endFace = startFace + nFaces
            cellStartFace = self.nInternalCells + startFace - self.nInternalFaces
            cellEndFace = self.nInternalCells + endFace - self.nInternalFaces

            if patch['type'] == 'processorCyclic':
                self.cellCentres[cellStartFace:cellEndFace] += self.faceCentres[startFace:endFace]

        return nLocalCells

    def createPaddedMesh(self):
        logger.info('generated padded mesh')
        if parallel.nProcessors == 1:
            return self
        mesh = Mesh()
        # set correct values for faces whose neighbours are ghost cells
        nLocalBoundaryFaces = self.nLocalCells - self.nInternalCells
        nLocalRemoteBoundaryFaces = self.nCells - self.nLocalCells
        nLocalFaces = self.nInternalFaces + nLocalBoundaryFaces

        exchanger = Exchanger()
        # processor patches are in increasing order
        remoteInternal = {'mapping':{},'owner':{}, 'neighbour':{}, 'areas':{}, 'weights':{}, 'normals':{}, 'volumes':{}}
        remoteBoundary = copy.deepcopy(remoteInternal)
        mesh.localRemoteCells = {'internal':{}, 'boundary':{}}
        mesh.localRemoteFaces = copy.deepcopy(mesh.localRemoteCells)
        mesh.remoteCells = copy.deepcopy(mesh.localRemoteCells)
        mesh.remoteFaces = copy.deepcopy(mesh.localRemoteCells)
        for patchID in self.remotePatches:
            patch = self.boundary[patchID]
            startFace = patch['startFace']
            nFaces = patch['nFaces']
            endFace = startFace + nFaces
            cellStartFace = self.nInternalCells + startFace - nLocalFaces
            cellEndFace = self.nInternalCells + endFace - nLocalFaces
            
            # extraInternalCells might not be unique
            #extraInternalCells = np.unique(self.owner[startFace:endFace])
            extraInternalCells = self.owner[startFace:endFace]
            extraFaces = self.cellFaces[extraInternalCells].ravel()
            boundaryFaces = range(startFace, endFace)
            extraFaces  = np.setdiff1d(extraFaces, boundaryFaces)
            owner = self.owner[extraFaces]
            neighbour = self.neighbour[extraFaces]
            normals = self.normals[extraFaces]
            weights = self.weights[extraFaces]
            extraCells = np.concatenate((owner, neighbour))
            extraGhostCells = np.setdiff1d(extraCells, extraInternalCells)
            # check cells on another processor
            extraRemoteGhostCells = extraGhostCells[extraGhostCells >= self.nLocalCells]
            if len(extraRemoteGhostCells) > 0:
                print 'Extra remote ghost cells:', patchID, len(extraRemoteGhostCells)

            boundaryIndex = np.in1d(neighbour, extraGhostCells)
            # swap extra boundary faces whose owner is wrong
            swapIndex = np.in1d(owner, extraGhostCells)
            tmp = neighbour[swapIndex]
            neighbour[swapIndex] = owner[swapIndex]
            owner[swapIndex] = tmp
            ## flip normals and invert weights
            normals[swapIndex] *= -1
            weights[swapIndex] = 1-weights[swapIndex]

            boundaryIndex = np.in1d(neighbour, extraGhostCells)
            internalIndex = np.invert(boundaryIndex)
            extraBoundaryFaces = extraFaces[boundaryIndex]
            extraInternalFaces = extraFaces[internalIndex]

            local, remote, tag = self.getProcessorPatchInfo(patchID)
            tag = {0:tag}
            tagIncrement = len(self.origPatches) + 1

            def remoteExchange(field, sendData, location):
                order = 'C'
                if location == 'internal':
                    size = (sendData.shape[0]*2, ) + sendData.shape[1:]
                    remoteInternal[field][patchID] = np.zeros(size, sendData.dtype, order)
                    #print field, sendData.flags['F_CONTIGUOUS'], remoteInternal[field][patchID].flags['F_CONTIGUOUS']
                    exchanger.exchange(remote, sendData, remoteInternal[field][patchID], tag[0])
                else:
                    size = (sendData.shape[0]*4, ) + sendData.shape[1:]
                    remoteBoundary[field][patchID] = np.zeros(size, sendData.dtype, order)
                    exchanger.exchange(remote, sendData, remoteBoundary[field][patchID], tag[0])
                tag[0] += tagIncrement


            #0: send extraInternalCells, first layer mapping
            remoteExchange('mapping', extraInternalCells, 'internal')

            #2: send extraGhostCells, second layer mapping
            remoteExchange('mapping', extraGhostCells, 'boundary')

            #4: send owner and neighbour and do the mapping
            remoteExchange('owner', owner[internalIndex], 'internal')
            remoteExchange('owner', owner[boundaryIndex], 'boundary')
            remoteExchange('neighbour', neighbour[internalIndex], 'internal')
            remoteExchange('neighbour', neighbour[boundaryIndex], 'boundary')

            #12: send rest
            remoteExchange('areas', self.areas[extraInternalFaces], 'internal')
            remoteExchange('areas', self.areas[extraBoundaryFaces], 'boundary')
            # WHY THE FUCK IS normals F_CONTIGUOUS
            remoteExchange('normals', normals[internalIndex], 'internal')
            remoteExchange('normals', normals[boundaryIndex], 'boundary')
            remoteExchange('weights', weights[internalIndex], 'internal')
            remoteExchange('weights', weights[boundaryIndex], 'boundary')
            remoteExchange('volumes', self.volumes[extraInternalCells], 'internal')
            
            mesh.localRemoteCells['internal'][patchID] = extraInternalCells
            mesh.localRemoteCells['boundary'][patchID] = extraGhostCells
            mesh.localRemoteFaces['internal'][patchID] = extraInternalFaces
            mesh.localRemoteFaces['boundary'][patchID] = extraBoundaryFaces
        statuses = exchanger.wait()

        getCount = lambda index: statuses[index].Get_count()/4
        nRemoteInternalFaces = 0
        nRemoteBoundaryFaces = 0
        total = 26
        for index, patchID in enumerate(self.remotePatches):
            remoteInternal['mapping'][patchID] = remoteInternal['mapping'][patchID][:getCount(index*total + 1)]
            remoteBoundary['mapping'][patchID] = remoteBoundary['mapping'][patchID][:getCount(index*total + 3)]
            remoteInternal['owner'][patchID] = remoteInternal['owner'][patchID][:getCount(index*total + 5)]
            remoteBoundary['owner'][patchID] = remoteBoundary['owner'][patchID][:getCount(index*total + 7)]
            mesh.remoteCells['internal'][patchID] = remoteInternal['mapping'][patchID]
            mesh.remoteCells['boundary'][patchID] = remoteBoundary['mapping'][patchID]
            mesh.remoteFaces['internal'][patchID] = len(remoteInternal['owner'][patchID])
            mesh.remoteFaces['boundary'][patchID] = len(remoteBoundary['owner'][patchID])
            #print(getCount(index*total+3))
            #print(parallel.rank, remoteBoundary['mapping'][patchID])
            nRemoteInternalFaces += len(remoteInternal['owner'][patchID])
            nRemoteBoundaryFaces += len(remoteBoundary['owner'][patchID])

        mesh.nInternalFaces = self.nInternalFaces + nLocalRemoteBoundaryFaces + nRemoteInternalFaces
        mesh.nBoundaryFaces = nLocalBoundaryFaces + nRemoteBoundaryFaces
        mesh.nFaces = mesh.nInternalFaces + mesh.nBoundaryFaces
        mesh.nInternalCells = self.nInternalCells + nLocalRemoteBoundaryFaces
        mesh.nGhostCells = mesh.nBoundaryFaces
        mesh.nCells = mesh.nInternalCells + mesh.nGhostCells

        nLocalInternalFaces = self.nInternalFaces + nLocalRemoteBoundaryFaces
        remoteGhostStartFace = mesh.nInternalFaces + nLocalBoundaryFaces
        def padFaceField(fieldName):
            field = getattr(self, fieldName)
            example = remoteInternal[fieldName][self.remotePatches[0]]
            size = (mesh.nFaces, ) + example.shape[1:]
            dtype = example.dtype

            faceField = np.empty(size, dtype)
            faceField[:self.nInternalFaces] = field[:self.nInternalFaces]
            faceField[self.nInternalFaces:nLocalInternalFaces] = field[nLocalFaces:]
            faceField[mesh.nInternalFaces:remoteGhostStartFace] = field[self.nInternalFaces:nLocalFaces]
            internalCursor = nLocalInternalFaces
            boundaryCursor = remoteGhostStartFace
            for patchID in self.remotePatches:
                nInternalFaces = len(remoteInternal['owner'][patchID])
                faceField[internalCursor:internalCursor + nInternalFaces] = remoteInternal[fieldName][patchID][:nInternalFaces]
                internalCursor += nInternalFaces
                nBoundaryFaces = len(remoteBoundary['owner'][patchID])
                faceField[boundaryCursor:boundaryCursor + nBoundaryFaces] = remoteBoundary[fieldName][patchID][:nBoundaryFaces]
                boundaryCursor += nBoundaryFaces
            return faceField

        mesh.owner = padFaceField('owner')
        mesh.neighbour = padFaceField('neighbour')
        mesh.neighbour[self.nInternalFaces:nLocalInternalFaces] -= nLocalBoundaryFaces
        mesh.neighbour[mesh.nInternalFaces:remoteGhostStartFace] += nLocalRemoteBoundaryFaces
        # do mapping, vectorize? mostly not possible
        internalCursor = nLocalInternalFaces
        boundaryCursor = remoteGhostStartFace
        internalCellsCursor = self.nInternalCells
        boundaryCellsCursor = self.nCells
        for patchID in self.remotePatches:
            reverseInternalMapping = {v:(k + internalCellsCursor) for k,v in enumerate(remoteInternal['mapping'][patchID])}
            nInternalFaces = len(remoteInternal['owner'][patchID])
            for index in range(internalCursor, internalCursor + nInternalFaces):
                mesh.owner[index] = reverseInternalMapping[mesh.owner[index]]
                mesh.neighbour[index] = reverseInternalMapping[mesh.neighbour[index]]
            internalCursor += nInternalFaces
            internalCellsCursor += len(remoteInternal['mapping'][patchID])
            reverseBoundaryMapping = {v:(k + boundaryCellsCursor) for k,v in enumerate(remoteBoundary['mapping'][patchID])}
            nBoundaryFaces = len(remoteBoundary['owner'][patchID])
            for index in range(boundaryCursor, boundaryCursor + nBoundaryFaces):
                mesh.owner[index] = reverseInternalMapping[mesh.owner[index]]
                mesh.neighbour[index] = reverseBoundaryMapping[mesh.neighbour[index]]
            boundaryCursor += nBoundaryFaces
            boundaryCellsCursor += len(remoteBoundary['mapping'][patchID])
     
        mesh.areas = padFaceField('areas')
        mesh.normals = padFaceField('normals')
        #print sum(np.linalg.norm(mesh.normals[remoteGhostStartFace:], axis=1)), nRemoteBoundaryFaces
        mesh.weights = padFaceField('weights')
        mesh.sumOp = self.getSumOp(mesh)

        mesh.volumes = np.empty((mesh.nInternalCells, 1), config.precision)
        mesh.volumes[:self.nInternalCells] = self.volumes
        internalCursor = self.nInternalCells
        for patchID in self.remotePatches:
            nInternalCells = self.boundary[patchID]['nFaces']
            mesh.volumes[internalCursor:internalCursor + nInternalCells] = remoteInternal['volumes'][patchID][:nInternalCells]
            internalCursor += nInternalCells

        return mesh

def removeCruft(content, keepHeader=False):
    # remove comments and newlines
    content = re.sub(re.compile('/\*.*\*/',re.DOTALL ) , '' , content)
    content = re.sub(re.compile('//.*\n' ) , '' , content)
    content = re.sub(re.compile('\n\n' ) , '\n' , content)
    # remove header
    if not keepHeader:
        content = re.sub(re.compile('FoamFile\n{(.*?)}\n', re.DOTALL), '', content)
    return content


def extractField(data, size, vector):
    extractScalar = lambda x: re.findall('[0-9\.Ee\-]+', x)
    if vector:
        extractor = lambda y: list(map(extractScalar, re.findall('\(([0-9\.Ee\-\r\n\s\t]+)\)', y)))
    else:
        extractor = extractScalar
    nonUniform = re.search('nonuniform', data)
    data = re.search(re.compile('[A-Za-z<>\s\r\n]+(.*)', re.DOTALL), data).group(1)
    if nonUniform is not None:
        start = data.find('(') + 1
        end = data.rfind(')')
        if config.fileFormat == 'binary':
            internalField = np.array(np.fromstring(data[start:end], dtype=np.float64))
            if vector:
                internalField = internalField.reshape((len(internalField)/3, 3))
        else:
            internalField = np.array(np.array(extractor(data[start:end]), dtype=np.float64))
        if not vector:
            internalField = internalField.reshape((-1, 1))
    else:
        internalField = np.array(np.tile(np.array(extractor(data)), (size, 1)), dtype=np.float64)
    return internalField.astype(config.precision)

def writeField(handle, field, dtype, initial):
    handle.write(initial + ' nonuniform List<'+ dtype +'>\n')
    handle.write('{0}\n('.format(len(field)))
    if config.fileFormat == 'binary':
        handle.write(ad.value(field.astype(np.float64)).tostring())
    else:
        handle.write('\n')
        for value in ad.value(field):
            if dtype == 'scalar':
                handle.write(str(value[0]) + '\n')
            else:
                handle.write('(' + ' '.join(np.char.mod('%f', value)) + ')\n')
    handle.write(')\n;\n')

