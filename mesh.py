import numpy as np
import scipy as sp
from scipy import sparse as sparse
import h5py
import re
import time
import copy
import os

import config, parallel
from config import ad, adsparse, T
from compat import norm, decompose
from parallel import pprint, Exchanger
from compat import printMemUsage, getCells

logger = config.Logger(__name__)

class Mesh(object):
    constants = ['nInternalFaces',
                 'nFaces',
                 'nBoundaryFaces',
                 'nInternalCells',
                 'nGhostCells',
                 'nCells',
                 'nLocalCells']
                 
    fields = ['owner', 'neighbour',
              'areas', 'volumes',
              'weights', 'deltas', 'normals',
              'cellCentres', 'faceCentres',
              'sumOp']

    def __init__(self):
        for attr in Mesh.constants:
            setattr(self, attr, 0)
        for attr in Mesh.fields:
            setattr(self, attr, np.array([[]]))
        self.boundary = []

        #self.localRemoteCells = None
        #self.localRemoteFaces = None
        #self.remoteCells = None
        #self.remoteFaces = None

    @classmethod
    def copy(cls, mesh, constants=True, fields=False):
        self = cls()
        self.boundary = copy.deepcopy(mesh.boundary)
        if fields:
            for attr in cls.fields:
              setattr(self, attr, copy.deepcopy(getattr(mesh, attr)))
        if constants:
            for attr in cls.constants:
              setattr(self, attr, getattr(mesh, attr))
        return self

    @classmethod
    def create(cls, caseDir=None, currTime='constant'):
        start = time.time()
        pprint('Reading mesh')
        self = cls()

        if config.hdf5:
            self.readHDF5(caseDir)
        else:
            self.readFoam(caseDir, currTime) 

        # patches
        localPatches, self.remotePatches = self.splitPatches(self.boundary)
        self.boundaryTensor = {}
        self.origPatches = copy.copy(localPatches)
        self.origPatches.sort()
        self.defaultBoundary = self.getDefaultBoundary()
        self.calculatedBoundary = self.getCalculatedBoundary()

        start = time.time()

        self.computeBeforeWrite()

       
        self.normals = self.getNormals()
        self.faceCentres, self.areas = self.getFaceCentresAndAreas()
        self.cellCentres, self.volumes = self.getCellCentresAndVolumes() # nCells after ghost cell mod
        # uses neighbour
        self.sumOp = self.getSumOp(self)             # (nInternalCells, nFaces)
        
        # ghost cell modification
        self.nLocalCells = self.createGhostCells()
        self.deltas = self.getDeltas()           # nFaces 
        self.weights = self.getWeights()   # nFaces


        # theano shared variables
        self.origMesh = cls.copy(self, fields=True)
        # update mesh initialization call
        self.update(currTime, 0.)
        self.makeTensor()

        pprint('nCells:', parallel.sum(self.origMesh.nInternalCells))
        pprint('nFaces:', parallel.sum(self.origMesh.nFaces))
        parallel.mpi.Barrier()
        end = time.time()
        pprint('Time for reading mesh:', end-start)
        pprint()

        printMemUsage()

        return self

    def getTimeDir(self, time):
        if time.is_integer():
            time = int(time)
        return '{0}/{1}/'.format(self.case, time)

    # re-reading after mesh creation
    def read(self, time):
        pprint('Re-reading mesh, time', time)
        timeDir = self.getTimeDir(time) 
        meshDir = timeDir + 'polyMesh/'
        # HACK correct updating
        boundary = self.readFoamBoundary(meshDir + 'boundary')
        for patchID in boundary:
            if boundary[patchID]['type'] == 'slidingPeriodic1D':
                self.origMesh.boundary[patchID].update(boundary[patchID])
        self.update(time, 0.)

    def readFoam(self, caseDir, currTime):
        pprint('reading foam mesh')
        self.case = caseDir + parallel.processorDirectory
        if isinstance(currTime, float):
            timeDir = self.getTimeDir(currTime)
        else:
            timeDir = self.case + currTime + '/'

        meshDir = timeDir + 'polyMesh/'
        self.faces = self.readFoamFile(meshDir + 'faces', np.int32)
        self.points = self.readFoamFile(meshDir + 'points', np.float64).astype(config.precision)
        self.owner = self.readFoamFile(meshDir + 'owner', np.int32).ravel()
        self.neighbour = self.readFoamFile(meshDir + 'neighbour', np.int32).ravel()
        
        self.boundary = self.readFoamBoundary(meshDir + 'boundary')

    def readFoamFile(self, foamFile, dtype):
        logger.info('read {0}'.format(foamFile))
        try: 
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
                    #print content[endIndices:startData+1]
                    data = np.fromstring(content[startData:end], dtype)
                    nFacePoints = faceIndices[0] 
                    return np.hstack((faceIndices.reshape(-1, 1), data.reshape(len(data)/nFacePoints, nFacePoints)))
                else:
                    data = np.fromstring(content[start:end], dtype)
                    if foamFile[-6:] == 'points':
                        data = data.reshape(len(data)/3, 3)
                    return data
            else:
                f = lambda x: list(filter(None, re.split('[ ()\n]+', x)))
                return np.array(list(map(f, filter(None, re.split('\n', content[start:end])))), dtype)
        except Exception as e: 
            config.exceptInfo(e, foamFile)


    def splitPatches(self, boundary):
        localPatches = []
        remotePatches = []

        for patchID in boundary.keys():
            if boundary[patchID]['type'] in config.processorPatches:
                remotePatches.append(patchID)
            else:
                localPatches.append(patchID)
        return localPatches, remotePatches

    def readFoamBoundary(self, boundaryFile):
        logger.info('read {0}'.format(boundaryFile))
        try:
            content = removeCruft(open(boundaryFile).read())
            patches = re.findall(re.compile('([A-Za-z0-9_]+)[\r\s\n\t]+{(.*?;[\r\s\n\t]+)}[\r\s\n\t]+', re.DOTALL), content)
        except Exception as e: 
            config.exceptInfo(e, boundaryFile)

        boundary = {}
        for patch in patches:
            try:
                boundary[patch[0]] = dict(re.findall('\n[ \t]+([a-zA-Z]+)[ ]+(.*?);', patch[1]))
                #nonuniform = re.findall('\n[ \t]+([a-zA-Z]+)[ ]+(nonuniform[ ]+List<([a-z]+)+[\r\s\n\t ]+([0-9]+).*?);[\r\s\n]+', patch[1], re.DOTALL)
                # HACK
                nonuniform = re.findall('\n[ \t]+([a-zA-Z]+)[ ]+(nonuniform.*?\))\n;\n', patch[1], re.DOTALL)
            except Exception as e: 
                config.exceptInfo(e, (boundaryFile, patch[0], patch[1]))

            for field in nonuniform:
                #print patch[0], field[0], len(field[1])
                boundary[patch[0]][field[0]] = field[1]
            boundary[patch[0]]['nFaces'] = int(boundary[patch[0]]['nFaces'])
            boundary[patch[0]]['startFace'] = int(boundary[patch[0]]['startFace'])
        return boundary

    def readHDF5(self, caseDir):
        pprint('reading hdf5 mesh')

        self.case = caseDir 
        meshFile = h5py.File(self.case + 'mesh.hdf5', 'r', driver='mpio', comm=parallel.mpi)
        assert meshFile['parallel/start'].shape[0] == parallel.nProcessors

        rank = parallel.rank
        parallelStart = meshFile['parallel/start'][rank]
        parallelEnd = meshFile['parallel/end'][rank]
        self.faces = np.array(meshFile['faces'][parallelStart[0]:parallelEnd[0]])
        self.points = np.array(meshFile['points'][parallelStart[1]:parallelEnd[1]])
        self.owner = np.array(meshFile['owner'][parallelStart[2]:parallelEnd[2]])
        self.neighbour = np.array(meshFile['neighbour'][parallelStart[3]:parallelEnd[3]])
        self.boundary = self.readHDF5Boundary(meshFile['boundary'][parallelStart[4]:parallelEnd[4]])
        #import pdb;pdb.set_trace()
        meshFile.close()

    def readHDF5Boundary(self, boundaryData):
        boundary = {}
        for patchID, key, value in boundaryData:
            if patchID not in boundary:
                boundary[patchID] = {}
            if key in ['nFaces', 'startFace']:
                value = int(value)
            boundary[patchID][key] = value
        return boundary

    def write(self, time):
        pprint('writing mesh, time', time)
        timeDir = self.getTimeDir(time) 
        meshDir = timeDir + 'polyMesh/'
        if not os.path.exists(meshDir):
            os.makedirs(meshDir)
        self.writeFoamBoundary(meshDir + 'boundary', self.origMesh.boundary)

    def writeFoamFile(self, fileName, data):
        assert config.fileFormat == 'binary'

        logger.info('writing {0}'.format(fileName))
        handle = open(fileName, 'w')
        handle.write(config.foamHeader)
        handle.write('FoamFile\n{\n')
        foamFile = config.foamFile.copy()
        foamFile['object'] = os.path.basename(fileName)
        if foamFile['object'] == 'points':
            foamFile['class'] = 'vectorField'
        elif foamFile['object'] == 'faces':
            foamFile['class'] = 'faceCompactList'
        else:
            foamFile['class'] = 'labelList'

        foamFile['location'] = 'constant/polyMesh'
        for key in foamFile:
            handle.write('\t' + key + ' ' + foamFile[key] + ';\n')
        handle.write('}\n')
        handle.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n')
        
        if foamFile['object'] == 'faces':
            nFaces = len(data)
            faceData = np.arange(0, nFaces*4 + 1, 4, np.int32)
            handle.write('{0}\n('.format(len(faceData)))
            handle.write(faceData.tostring())
            handle.write(')\n')

            handle.write('{0}\n('.format(np.prod(data.shape)))
            handle.write(data.tostring())
            handle.write(')\n')
        else:
            handle.write('{0}\n('.format(len(data)))
            handle.write(data.tostring())
            handle.write(')\n\n')

        handle.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
        handle.close()

    def writeFoamBoundary(self, boundaryFile, boundary):
        logger.info('writing {0}'.format(boundaryFile))
        handle = open(boundaryFile, 'w')
        handle.write(config.foamHeader)
        handle.write('FoamFile\n{\n')
        foamFile = config.foamFile.copy()
        foamFile['class'] = 'polyBoundaryMesh'
        foamFile['object'] = 'boundary'
        foamFile['location'] = 'constant/polyMesh'
        for key in foamFile:
            handle.write('\t' + key + ' ' + foamFile[key] + ';\n')
        handle.write('}\n')
        handle.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
        nPatches = len(boundary)
        handle.write(str(nPatches) + '\n(\n')
        patchIDs = boundary.keys()
        patchIDs = sorted(patchIDs, key=lambda x: (boundary[x]['startFace'], boundary[x]['nFaces']))
        for patchID in patchIDs:
            handle.write('\t' + patchID + '\n')
            handle.write('\t{\n')
            patch = boundary[patchID]
            for attr in patch:
                value = patch[attr]
                if attr.startswith('loc_'):
                    continue
                elif attr == 'transform':
                    value = 'unknown'
                if isinstance(value, np.ndarray):
                    writeField(handle, value, 'vector', '\t\t' + attr)
                else:
                    handle.write('\t\t{0} {1};\n'.format(attr, value))
            handle.write('\t}\n')
        handle.write(')\n')
        handle.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
        handle.close()

    def writeHDF5(self, case):
        pprint('writing hdf5 mesh')
        meshFile = h5py.File(case + 'mesh.hdf5', 'w', driver='mpio', comm=parallel.mpi)
        rank = parallel.rank
        nProcs = parallel.nProcessors

        mesh = self.origMesh
        boundary = []
        for patchID in mesh.boundary.keys():
            for key, value in mesh.boundary[patchID].iteritems():
                boundary.append([patchID, key, str(value)])
        boundary = np.array(boundary, dtype='S100')
        faces, points, owner, neighbour = self.faces, self.points, mesh.owner, mesh.neighbour
        cells = self.cells

        if mesh.nInternalFaces > 0:
            neighbour = neighbour[:mesh.nInternalFaces]

        parallelInfo = np.array([faces.shape[0], points.shape[0], \
                                 owner.shape[0], neighbour.shape[0],
                                 boundary.shape[0],
                                 cells.shape[0]
                                ])
        parallelStart = np.zeros_like(parallelInfo)
        parallelEnd = np.zeros_like(parallelInfo)
        parallel.mpi.Exscan(parallelInfo, parallelStart)
        parallel.mpi.Scan(parallelInfo, parallelEnd)
        parallelSize = parallelEnd.copy()
        parallel.mpi.Bcast(parallelSize, nProcs-1)

        parallelGroup = meshFile.create_group('parallel')
        parallelStartData = parallelGroup.create_dataset('start', (nProcs, len(parallelInfo)), np.int64)
        parallelStartData[rank] = parallelStart
        parallelEndData = parallelGroup.create_dataset('end', (nProcs, len(parallelInfo)), np.int64)
        parallelEndData[rank] = parallelEnd
        
        facesData = meshFile.create_dataset('faces', (parallelSize[0],) + faces.shape[1:], faces.dtype)
        facesData[parallelStart[0]:parallelEnd[0]] = faces
        pointsData = meshFile.create_dataset('points', (parallelSize[1],) + points.shape[1:], np.float64)
        pointsData[parallelStart[1]:parallelEnd[1]] = points.astype(np.float64)
        ownerData = meshFile.create_dataset('owner', (parallelSize[2],) + owner.shape[1:], owner.dtype)
        ownerData[parallelStart[2]:parallelEnd[2]] = owner
        neighbourData = meshFile.create_dataset('neighbour', (parallelSize[3],) + neighbour.shape[1:], neighbour.dtype)
        neighbourData[parallelStart[3]:parallelEnd[3]] = neighbour

        boundaryData = meshFile.create_dataset('boundary', (parallelSize[4], 3), 'S100') 
        boundaryData[parallelStart[4]:parallelEnd[4]] = boundary

        cellsData = meshFile.create_dataset('cells', (parallelSize[5],) + cells.shape[1:], cells.dtype)
        cellsData[parallelStart[5]:parallelEnd[5]] = cells

        meshFile.close()

    def computeBeforeWrite(self):
        self.populateSizes()
        # mesh computation
        # uses neighbour
        self.cellFaces = self.getCellFaces()     # nInternalCells
        # time consuming 
        self.cells = getCells(self)

    def populateSizes(self):
        self.nInternalFaces = len(self.neighbour)
        self.nFaces = len(self.owner)
        self.nBoundaryFaces = self.nFaces-self.nInternalFaces
        self.nInternalCells = np.max(self.owner)+1
        self.nGhostCells = self.nBoundaryFaces
        self.nCells = self.nInternalCells + self.nGhostCells

    def getNormals(self):
        logger.info('generated normals')
        v1 = self.points[self.faces[:,1]]-self.points[self.faces[:,2]]
        v2 = self.points[self.faces[:,2]]-self.points[self.faces[:,3]]
        # CROSS product makes it F_CONTIGUOUS even if normals is not
        normals = np.cross(v1, v2)
        # change back to contiguous
        normals = np.ascontiguousarray(normals)
        return normals / norm(normals, axis=1, keepdims=True)

    def getCellFaces(self):
        logger.info('generated cell faces') 
        enum = lambda x: np.column_stack((np.indices(x.shape, np.int32)[0], x)) 
        combined = np.concatenate((enum(self.owner), enum(self.neighbour)))
        cellFaces = combined[combined[:,1].argsort(), 0]
        # todo: make it a list ( investigate np.diff )
        return cellFaces.reshape(self.nInternalCells, len(cellFaces)/self.nInternalCells)

    def getCellCentresAndVolumes(self):
        logger.info('generated cell centres and volumes')
        nCellFaces = self.cellFaces.shape[1]
        cellCentres = np.mean(self.faceCentres[self.cellFaces], axis=1)
        sumCentres = cellCentres*0
        sumVolumes = np.sum(sumCentres, axis=1, keepdims=True)
        areaNormals = self.areas * self.normals
        for index in range(0, nCellFaces):
            indices = self.cellFaces[:,index]
            height = cellCentres-self.faceCentres[indices]
            volumes = np.abs(np.sum(areaNormals[indices]*height, axis=1, keepdims=True))/3
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
            areas = norm(normals, axis=1, keepdims=True)/2
            sumAreas += areas
            sumCentres += areas*centres
        faceCentres = sumCentres/sumAreas
        return faceCentres, sumAreas

    def getDeltas(self):
        logger.info('generated deltas')
        P = self.cellCentres[self.owner]
        N = self.cellCentres[self.neighbour]
        return norm(P-N, axis=1, keepdims=True)

    def getWeights(self):
        logger.info('generated face deltas')
        P = self.cellCentres[self.owner]
        N = self.cellCentres[self.neighbour]
        F = self.faceCentres
        neighbourDist = np.abs(np.sum((F-N)*self.normals, axis=1, keepdims=True))
        ownerDist = np.abs(np.sum((F-P)*self.normals, axis=1, keepdims=True))
        weights = neighbourDist/(neighbourDist + ownerDist)
        return weights

    def getSumOp(self, mesh, ghost=False):
        logger.info('generated sum op')
        owner = sparse.csc_matrix((np.ones(mesh.nFaces, config.precision), mesh.owner, np.arange(0, mesh.nFaces+1, dtype=np.int32)), shape=(mesh.nInternalCells, mesh.nFaces))
        Nindptr = np.concatenate((np.arange(0, mesh.nInternalFaces+1, dtype=np.int32), mesh.nInternalFaces*np.ones(mesh.nFaces-mesh.nInternalFaces, np.int32)))
        neighbour = sparse.csc_matrix((-np.ones(mesh.nInternalFaces, config.precision), mesh.neighbour[:mesh.nInternalFaces], Nindptr), shape=(mesh.nInternalCells, mesh.nFaces))
        # skip empty patches
        #for patchID in self.boundary:
        #    patch = self.boundary[patchID]
        #    if patch['type'] == 'empty' and patch['nFaces'] != 0:
        #        pprint('Deleting empty patch ', patchID)
        #        startFace = mesh.nInternalFaces + patch['startFace'] - self.nInternalFaces
        #        endFace = startFace + patch['nFaces']
        #        owner.data[startFace:endFace] = 0
        sumOp = (owner + neighbour).tocsr()
    
        #return adsparse.CSR(sumOp.data, sumOp.indices, sumOp.indptr, sumOp.shape)
        return sumOp

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
        rank = parallel.rank
        self.cellCentres = np.concatenate((self.cellCentres, np.zeros((self.nBoundaryFaces, 3), config.precision)))
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
            if patch['type'] in config.cyclicPatches:
                #print patchID, self.cellCentres[self.owner[startFace:endFace]][0]
                neighbourPatch = self.boundary[patch['neighbourPatch']]   
                neighbourStartFace = neighbourPatch['startFace']
                neighbourEndFace = neighbourStartFace + nFaces
                # apply transformation: single value
                # append cell centres
                patch['neighbourIndices'] = self.owner[neighbourStartFace:neighbourEndFace]
                patch['transform'] = self.faceCentres[startFace]-self.faceCentres[neighbourStartFace]
                self.cellCentres[cellStartFace:cellEndFace] = patch['transform'] + self.cellCentres[patch['neighbourIndices']]
            elif patch['type'] == 'processor':
                patch['neighbProcNo'] = int(patch['neighbProcNo'])
                patch['myProcNo'] = int(patch['myProcNo'])
                local, remote, tag = self.getProcessorPatchInfo(patchID)
                # exchange data
                exchanger.exchange(remote, self.cellCentres[self.owner[startFace:endFace]], self.cellCentres[cellStartFace:cellEndFace], tag)
                tag += len(self.origPatches) + 1
                patch['neighbourIndices'] = self.neighbour[startFace:endFace].copy()
                exchanger.exchange(remote, self.owner[startFace:endFace], patch['neighbourIndices'], tag)

            elif patch['type'] == 'processorCyclic':
                patch['neighbProcNo'] = int(patch['neighbProcNo'])
                patch['myProcNo'] = int(patch['myProcNo'])
                local, remote, tag = self.getProcessorPatchInfo(patchID)
                # apply transformation
                exchanger.exchange(remote, -self.faceCentres[startFace:endFace] + self.cellCentres[self.owner[startFace:endFace]], self.cellCentres[cellStartFace:cellEndFace], tag)
                tag += len(self.origPatches) + 1
                patch['neighbourIndices'] = self.neighbour[startFace:endFace].copy()
                exchanger.exchange(remote, self.owner[startFace:endFace], patch['neighbourIndices'], tag)
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

        
    def makeTensor(self):
        logger.info('making tensor variables')
        for attr in Mesh.constants:
            setattr(self, attr, ad.iscalar())
        for attr in Mesh.fields:
            value = getattr(self, attr) 
            if attr in ['owner', 'neighbour']:
                setattr(self, attr, ad.ivector())
            elif attr == 'sumOp':
                setattr(self, attr, adsparse.csr_matrix(dtype=config.dtype))
            elif value.shape[1] == 1:
                setattr(self, attr, ad.bcmatrix())
            else:
                setattr(self, attr, ad.matrix())

        for patchID in self.origPatches:
            for attr in self.getBoundaryTensor(patchID):
                self.boundary[patchID][attr[0]] = attr[1]


    def getBoundaryTensor(self, patchID):
        default = [('startFace', ad.iscalar()), ('nFaces', ad.iscalar())]
        return default + self.boundaryTensor.get(patchID, [])

    def update(self, t, dt):
        logger.info('updating mesh')
        start = time.time()
        mesh = self.origMesh
        for patchID in self.origPatches:
            patch = mesh.boundary[patchID]
            startFace = patch['startFace']
            endFace = startFace + patch['nFaces']
            if patch['type'] == 'slidingPeriodic1D':
                # single processor has everything
                if patch['nFaces'] == 0:
                    if dt == 0.:
                        self.boundaryTensor[patchID] = [('loc_multiplier', adsparse.csr_matrix(dtype=config.dtype))]
                    patch['loc_multiplier'] = sparse.csr_matrix((0,0), dtype=config.precision)
                    patch['loc_velocity'] = np.fromstring(patch['velocity'][1:-1], sep=' ', dtype=config.precision)
                    continue
                if dt == 0.:
                    patch['nLayers'] = int(patch['nLayers'])
                    patch['nFacesPerLayer'] = patch['nFaces']/patch['nLayers']
                    neighbourPatch = mesh.boundary[patch['neighbourPatch']]   
                    neighbourStartFace = neighbourPatch['startFace']
                    neighbourEndFace = neighbourStartFace + patch['nFacesPerLayer']
                    patch['loc_velocity'] = np.fromstring(patch['velocity'][1:-1], sep=' ', dtype=config.precision)
                    patch['loc_fixedCellCentres'] = mesh.cellCentres[mesh.owner[neighbourStartFace:neighbourEndFace]]
                    dists = sp.spatial.distance.squareform(sp.spatial.distance.pdist(patch['loc_fixedCellCentres']))
                    # is this guaranteed to give extreme indices?
                    index1, index2 = np.unravel_index(np.argmax(dists), dists.shape)
                    patch1 = mesh.boundary[patch['periodicPatch']]
                    patch2 = mesh.boundary[patch1['neighbourPatch']]
                    if np.linalg.norm(patch['loc_fixedCellCentres'][index1] + patch2['transform'] - patch['loc_fixedCellCentres'][index2]) > np.max(dists):
                        index2, index1 = index1, index2
                    point1 = patch['loc_fixedCellCentres'][index2] + patch1['transform']
                    point2 = patch['loc_fixedCellCentres'][index1] + patch2['transform']
                    #print patchID, mesh.cellCentres[mesh.owner[startFace:endFace]][0], point1, patch['velocity']
                    # reread
                    if 'movingCellCentres' in patch:
                        patch['movingCellCentres'] = extractField(patch['movingCellCentres'], patch['nFacesPerLayer']+1, (3,))
                    else:
                        patch['movingCellCentres'] = np.vstack((patch['loc_fixedCellCentres'].copy(), point2))
                    patch['loc_periodicLimit'] = point1
                    patch['loc_extraIndex'] = index1
                    self.boundaryTensor[patchID] = [('loc_multiplier', adsparse.csr_matrix(dtype=config.dtype))]
                patch['movingCellCentres'] += patch['loc_velocity']*dt
                # only supports low enough velocities
                transformIndices = (patch['movingCellCentres']-patch['loc_periodicLimit']).dot(patch['loc_velocity']) > 1e-6
                #print patchID, patch['movingCellCentres'][0], patch['loc_periodicLimit'], patch['loc_velocity']
                patch['movingCellCentres'][transformIndices] += mesh.boundary[mesh.boundary[patch['periodicPatch']]['neighbourPatch']]['transform']
                dists = sp.spatial.distance.cdist(patch['loc_fixedCellCentres'], patch['movingCellCentres'])
                n, m = dists.shape
                sortedDists = np.argsort(dists, axis=1)[:,:2]
                np.place(sortedDists, sortedDists == (m-1), patch['loc_extraIndex'])
                indices = np.arange(n).reshape(-1,1)
                minDists = dists[indices, sortedDists]
                # weights should use weighted average?
                weights = (minDists/minDists.sum(axis=1, keepdims=True))[:,[1, 0]].ravel()
                sortedDists = sortedDists.ravel()
                indices = np.hstack((indices, indices)).ravel()
                # repetition
                weights = np.tile(weights, patch['nLayers'])
                repeater = np.repeat(np.arange(patch['nLayers']), 2*patch['nFacesPerLayer'])*patch['nFacesPerLayer']
                indices = np.tile(indices, patch['nLayers']) + repeater
                sortedDists = np.tile(sortedDists, patch['nLayers']) + repeater
                loc_multiplier = sparse.coo_matrix((weights, (indices, sortedDists)), shape=(patch['nFaces'], patch['nFaces'])).tocsr()
                if 'loc_multiplier' not in patch:
                    patch['loc_multiplier'] = loc_multiplier
                else:
                    patch['loc_multiplier'].data = loc_multiplier.data
                    patch['loc_multiplier'].indices = loc_multiplier.indices
                    patch['loc_multiplier'].indptr = loc_multiplier.indptr
                #print patch['movingCellCentres']
                #print 'set', id(patch['loc_multiplier'].data)
        parallel.mpi.Barrier()
        end = time.time()
        pprint('Time to update mesh:', end-start)

    def decompose(self, nprocs):
        assert parallel.nProcessors == 1
        start = time.time()
        pprint('decomposing mesh')
        decomposed = decompose(self, nprocs)
        for n in range(0, nprocs):
            pprint('writing processor{}'.format(n))
            points, faces, owner, neighbour, boundary = decomposed[n]
            meshCase = self.case + 'processor{}/constant/polyMesh/'.format(n)
            if not os.path.exists(meshCase):
                os.makedirs(meshCase)
            self.writeFoamFile(meshCase + 'points', points)
            self.writeFoamFile(meshCase + 'faces', faces)
            self.writeFoamFile(meshCase + 'owner', owner)
            self.writeFoamFile(meshCase + 'neighbour', neighbour)
            self.writeFoamBoundary(meshCase + 'boundary', boundary)
        pprint('Time for decomposing mesh:', time.time()-start)
        pprint()
        return 

def removeCruft(content):
    header = re.search('FoamFile', content)
    content = content[header.start():]
    content = re.sub(re.compile(r'FoamFile\n{(.*?)}\n', re.DOTALL), '', content)
    # assume no number in comments
    begin = re.search('[0-9]+', content)
    content = content[begin.start():]
    # assume no brackets in comments
    end = content.rfind(')')
    content = content[:end+1]
    return content

def extractScalar(data):
    return re.findall('[0-9\.Ee\-]+', data)

def extractVector(data):
    return list(map(extractScalar, re.findall('\(([0-9\.Ee\-\r\n\s\t]+)\)', data)))

def extractField(data, size, dimensions):
    if isinstance(data, np.ndarray):
        assert data.shape == (size, ) + dimensions
        return data
    if size == 0:
        return np.zeros((0,) + dimensions, config.precision)
    if dimensions == (3,):
        extractor = extractVector
    else:
        extractor = extractScalar
    nonUniform = re.search('nonuniform', data)
    data = re.search(re.compile('[A-Za-z<>\s\r\n]+(.*)', re.DOTALL), data).group(1)
    if nonUniform is not None:
        start = data.find('(') + 1
        end = data.rfind(')')
        if start == end:
            internalField = np.zeros((size, ) + dimensions)
        elif config.fileFormat == 'binary':
            internalField = np.array(np.fromstring(data[start:end], dtype=np.float64))
        else:
            internalField = np.array(np.array(extractor(data[start:end]), dtype=np.float64))
    else:
        internalField = np.array(np.tile(np.array(extractor(data)), (size, 1)), dtype=np.float64)
    internalField = internalField.reshape((size, ) + dimensions)
    return internalField.astype(config.precision)

def writeField(handle, field, dtype, initial):
    handle.write(initial + ' nonuniform List<'+ dtype +'>\n')
    handle.write('{0}\n('.format(len(field)))
    if config.fileFormat == 'binary':
        handle.write(field.astype(np.float64).tostring())
    else:
        handle.write('\n')
        for value in field:
            if dtype == 'scalar':
                handle.write(str(value[0]) + '\n')
            else:
                handle.write('(' + ' '.join(np.char.mod('%f', value)) + ')\n')
    handle.write(')\n;\n')



