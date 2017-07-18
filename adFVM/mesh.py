import numpy as np
import scipy as sp
from scipy import sparse as sparse
import re
import time
import copy
import os

from . import config, parallel
from .compat import norm, decompose, getCells, add_at
from .memory import printMemUsage
from .parallel import pprint, Exchanger
from .tensor import Tensor, IntegerScalar

try:
    import h5py
except:
    pprint('h5py library not loaded')

logger = config.Logger(__name__)

class Mesh(object):
    intFields = ['owner', 'neighbour']
    gradFields = ['areas', 'volumesL', 'volumesR',
                  'weights', 'deltas', 'normals',
                  'linearWeights', 'quadraticWeights',
                  #'cellCentres', 'faceCentres', 
                  #'gradOp'
                 ]

    constants = ['nCells', 'nFaces', 'nInternalCells', 'nInternalFaces',
                 'nLocalCells']

    def __init__(self):
        pass
        #for attr in Mesh.constants:
        #    setattr(self, attr, 0)
        #for attr in Mesh.fields:
        #    setattr(self, attr, np.array([[]]))
        #self.boundary = []
        for attr in Mesh.gradFields:
            setattr(self, attr, 0)

        #self.localRemoteCells = None
        #self.localRemoteFaces = None
        #nameself.remoteCells = None
        #self.remoteFaces = None

    @classmethod
    def copy(cls, mesh, constants=True, fields=False, link=False):
        self = cls()
        self.boundary = copy.deepcopy(mesh.boundary)
        if fields:
            for attr in cls.fields:
                if link:
                    setattr(self, attr, getattr(mesh, attr))
                else:
                    setattr(self, attr, copy.deepcopy(getattr(mesh, attr)))
        if constants:
            for attr in cls.constants:
                setattr(self, attr, getattr(mesh, attr))
        self.parent = mesh
        return self

    @classmethod
    def create(cls, caseDir=None, currTime='constant'):
        self = cls()
        self.caseDir = caseDir
        if config.hdf5:
            meshData = self.readHDF5(caseDir)
        else:
            meshData = self.readFoam(caseDir, currTime) 

        self.build(meshData, currTime)

        return self

    @config.timeFunction('Time for building mesh')
    def build(self, meshData, currTime='constant'):
        pprint('Building mesh')

        self.points, self.faces, self.owner, self.neighbour, \
                self.addressing, self.boundary = meshData

        self.buildBeforeWrite()

        # patches
        self.localPatches, self.remotePatches = self.splitPatches(self.boundary)
        self.patches = self.localPatches + self.remotePatches
        self.nLocalPatches = len(self.localPatches)
        self.nRemotePatches = len(self.remotePatches)
        self.sortedPatches = copy.copy(self.localPatches)
        self.sortedPatches.sort()
        self.tags = {patchID: self.getProcessorPatchInfo(patchID)[2] for patchID in self.remotePatches}
        self.boundaryTensor = {}
        self.defaultBoundary = self.getDefaultBoundary()
        self.calculatedBoundary = self.getCalculatedBoundary()
       
        self.normals = self.getNormals()
        self.faceCentres, self.areas = self.getFaceCentresAndAreas()
        self.cellCentres, self.volumes = self.getCellCentresAndVolumes() 
        self.volumesL = self.volumes[self.owner]
        self.volumesR = self.volumes[self.neighbour[:self.nInternalFaces]]

        # ghost cell modification: neighbour and cellCentres
        self.nLocalCells = self.createGhostCells()
        self.deltas, self.deltasUnit = self.getDeltas()           # nFaces 
        self.weights, self.linearWeights, self.quadraticWeights = self.getWeights()   # nFaces

        # uses neighbour
        self.cellNeighboursMatOp = self.getCellNeighbours(boundary=False)
        self.cellNeighbours = self.getCellNeighbours()
        self.sumOp = self.getSumOp(self)             # (nInternalCells, nFaces)
        #self.gradOp = self.getGradOp(self)             # (nInternalCells, nCells)
        self.checkWeights()

        # theano shared variables
        self.origMesh = self
        self.symMesh = Mesh()
        # update mesh initialization call
        self.update(currTime, 0.)
        self.symMesh.parent = self
        self.symMesh.makeTensor()
        #import pdb;pdb.set_trace()

        pprint('nCells:', parallel.sum(self.origMesh.nInternalCells))
        pprint('nFaces:', parallel.sum(self.origMesh.nFaces))

        printMemUsage()
        return 

    def getTimeDir(self, time, case=None):
        if case is None:
            case = self.case
        if time.is_integer():
            time = int(time)
        return '{0}/{1}'.format(case, time)

    def getTimes(self):
        if config.hdf5:
            times = [float(x[:-5]) for x in os.listdir(self.case) if config.isfloat(x[:-5]) and x.endswith('.hdf5')]
        else:
            times = [float(x) for x in os.listdir(self.case) if config.isfloat(x) ]
        return sorted(times)

    def getFields(self, time):
        fields = os.listdir(self.getTimeDir(time))
        fields = filter(lambda x: x != 'polyMesh', fields)
        return fields

    # re-reading after mesh creation
    def read(self, timeDir):
        if config.hdf5:
            try:
                timeDir = timeDir['mesh']
            except KeyError:
                return
            boundary = self.readHDF5Boundary(timeDir)
        else:
            timeDir = timeDir + '/polyMesh/boundary'
            if os.path.exists(timeDir):
                boundary = self.readFoamBoundary(timeDir)
            else:
                return
        pprint('Re-reading mesh, time', timeDir)
        for patchID in boundary:
            if boundary[patchID]['type'] == 'slidingPeriodic1D':
                self.origMesh.boundary[patchID].update(boundary[patchID])
        self.update(time, 0.)

    @config.timeFunction('Time for reading mesh')
    def readFoam(self, caseDir, currTime):
        pprint('Reading foam mesh')
        self.case = caseDir + parallel.processorDirectory
        if isinstance(currTime, float):
            timeDir = self.getTimeDir(currTime) + '/'
        else:
            timeDir = self.case + currTime + '/'
        meshDir = timeDir + 'polyMesh/'
        constantMeshDir = self.case + 'constant/polyMesh/'

        faces = self.readFoamFile(constantMeshDir + 'faces', np.int32)
        pointsFile = meshDir + 'points'
        if not os.path.exists(pointsFile):
            pointsFile = constantMeshDir + 'points'
        points = self.readFoamFile(pointsFile, np.float64).astype(config.precision)
        owner = self.readFoamFile(constantMeshDir + 'owner', np.int32).ravel()
        neighbour = self.readFoamFile(constantMeshDir + 'neighbour', np.int32).ravel()
        addressing = []
        if os.path.exists(constantMeshDir + 'pointProcAddressing'):
            addressing.append(self.readFoamFile(constantMeshDir + 'pointProcAddressing', np.int32).ravel())
            addressing.append(self.readFoamFile(constantMeshDir + 'faceProcAddressing', np.int32).ravel())
            addressing.append(self.readFoamFile(constantMeshDir + 'cellProcAddressing', np.int32).ravel())
        
        boundaryFile = meshDir + 'boundary'
        if not os.path.exists(boundaryFile):
            boundaryFile = constantMeshDir + 'boundary'
        boundary = self.readFoamBoundary(boundaryFile)
        return points, faces, owner, neighbour, addressing, boundary

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
        patchIDs = boundary.keys()
        patchIDs = sorted(patchIDs, key=lambda x: (boundary[x]['startFace'], boundary[x]['nFaces']))
        for patchID in patchIDs:
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
            # cyclicAMI HACK
            if boundary[patch[0]]['type'] == 'cyclicAMI':
                boundary[patch[0]]['type'] = 'cyclic'
            boundary[patch[0]]['nFaces'] = int(boundary[patch[0]]['nFaces'])
            boundary[patch[0]]['startFace'] = int(boundary[patch[0]]['startFace'])
        return boundary

    @config.timeFunction('Time for reading mesh')
    def readHDF5(self, caseDir):
        pprint('Reading hdf5 mesh')

        self.case = caseDir 
        meshFile = h5py.File(self.case + 'mesh.hdf5', 'r', driver='mpio', comm=parallel.mpi)
        assert meshFile['parallel/start'].shape[0] == parallel.nProcessors

        rank = parallel.rank
        parallelStart = meshFile['parallel/start'][rank]
        parallelEnd = meshFile['parallel/end'][rank]
        faces = np.array(meshFile['faces'][parallelStart[0]:parallelEnd[0]])
        points = np.array(meshFile['points'][parallelStart[1]:parallelEnd[1]])
        owner = np.array(meshFile['owner'][parallelStart[2]:parallelEnd[2]])
        neighbour = np.array(meshFile['neighbour'][parallelStart[3]:parallelEnd[3]])
        addressing = []
        if parallel.nProcessors > 1:
            addressing.append(np.array(meshFile['pointProcAddressing'][parallelStart[5]:parallelEnd[5]]))
            addressing.append(np.array(meshFile['faceProcAddressing'][parallelStart[6]:parallelEnd[6]]))
            addressing.append(np.array(meshFile['cellProcAddressing'][parallelStart[7]:parallelEnd[7]]))
        boundary = self.readHDF5Boundary(meshFile)
        meshFile.close()

        return points, faces, owner, neighbour, addressing, boundary

    def readHDF5Boundary(self, meshFile):
        boundary = {}
        rank = parallel.rank
        boundaryGroup = meshFile['boundary']
        parallelStart = boundaryGroup['parallel/start'][rank]
        parallelEnd = boundaryGroup['parallel/end'][rank]
        boundaryData = boundaryGroup['values'][parallelStart[0]:parallelEnd[0]]

        for patchID, key, value in boundaryData:
            if patchID not in boundary:
                boundary[patchID] = {}
            if key in ['nFaces', 'startFace']:
                value = int(value)
            boundary[patchID][key] = value

        fieldGroup = boundaryGroup['fields']
        index = 1
        for pair in fieldGroup:
            # how to ensure order?
            patchID, key = pair.split('__')
            value = boundaryGroup['fields'][pair][parallelStart[index]:parallelEnd[index]]
            boundary[patchID][key] = value
            index += 1

        return boundary

    def write(self, timeDir):
        pprint('writing mesh, time', time)
        if config.hdf5:
            meshDir = timeDir.create_group('mesh')
            self.writeHDF5Boundary(meshDir)
        else:
            meshDir = timeDir + '/polyMesh/'
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

            handle.write('{0}\n('.format(data.shape[0]*4))
            handle.write(data[:,1:].tostring())
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

        mesh = self.origMesh
        faces, points, owner, neighbour = self.faces, self.points, mesh.owner, mesh.neighbour
        cells = self.cells

        if mesh.nInternalFaces > 0:
            neighbour = neighbour[:mesh.nInternalFaces]

        parallelInfo = np.array([faces.shape[0], points.shape[0], \
                                 owner.shape[0], neighbour.shape[0],
                                 cells.shape[0]
                                ])
        if parallel.nProcessors > 1:
            parallelInfo = np.concatenate((parallelInfo, [x.shape[0] for x in self.addressing]))

        parallelStart, parallelEnd, parallelSize = self.writeHDF5Parallel(meshFile, parallelInfo)

        facesData = meshFile.create_dataset('faces', (parallelSize[0],) + faces.shape[1:], faces.dtype)
        facesData[parallelStart[0]:parallelEnd[0]] = faces
        pointsData = meshFile.create_dataset('points', (parallelSize[1],) + points.shape[1:], np.float64)
        pointsData[parallelStart[1]:parallelEnd[1]] = points.astype(np.float64)
        ownerData = meshFile.create_dataset('owner', (parallelSize[2],) + owner.shape[1:], owner.dtype)
        ownerData[parallelStart[2]:parallelEnd[2]] = owner
        neighbourData = meshFile.create_dataset('neighbour', (parallelSize[3],) + neighbour.shape[1:], neighbour.dtype)
        neighbourData[parallelStart[3]:parallelEnd[3]] = neighbour

        cellsData = meshFile.create_dataset('cells', (parallelSize[4],) + cells.shape[1:], cells.dtype)
        cellsData[parallelStart[4]:parallelEnd[4]] = cells
        if parallel.nProcessors > 1:
            names = ['pointProcAddressing', 'faceProcAddressing', 'cellProcAddressing']
            for index in range(0, len(names)):
                parI = index + 5
                data = self.addressing[index]
                cellsData = meshFile.create_dataset(names[index], (parallelSize[parI],) + data.shape[1:], data.dtype)
                cellsData[parallelStart[parI]:parallelEnd[parI]] = data

        self.writeHDF5Boundary(meshFile)

        meshFile.close()

    def writeHDF5Parallel(self, meshFile, parallelInfo):
        rank = parallel.rank
        nProcs = parallel.nProcessors
        parallelStart = np.zeros_like(parallelInfo)
        parallelEnd = np.zeros_like(parallelInfo)
        parallel.mpi.Exscan(parallelInfo, parallelStart)
        parallel.mpi.Scan(parallelInfo, parallelEnd)
        parallelSize = parallelEnd.copy()
        parallel.mpi.Bcast(parallelSize, nProcs-1)

        parallelGroup = meshFile.require_group('parallel')
        parallelStartData = parallelGroup.require_dataset('start', (nProcs, len(parallelInfo)), np.int64)
        with parallelStartData.collective:
            parallelStartData[rank] = parallelStart
        parallelEndData = parallelGroup.require_dataset('end', (nProcs, len(parallelInfo)), np.int64)
        with parallelEndData.collective:
            parallelEndData[rank] = parallelEnd
        return parallelStart, parallelEnd, parallelSize

    def writeHDF5Boundary(self, meshFile):
        boundary = []
        boundaryField = []
        for patchID in self.patches:
            for key, value in self.origMesh.boundary[patchID].iteritems():
                if key.startswith('loc_'):
                    continue
                elif isinstance(value, np.ndarray):
                    boundaryField.append((patchID, key, value))
                else:
                    boundary.append([patchID, key, value])
        boundary = np.array(boundary, dtype='S100')

        parallelInfo = [boundary.shape[0]]
        for _, _, value in boundaryField:
            parallelInfo.append(value.shape[0])
        parallelInfo = np.array(parallelInfo)

        boundaryGroup = meshFile.create_group('boundary')
        parallelStart, parallelEnd, parallelSize = self.writeHDF5Parallel(boundaryGroup, parallelInfo)
        boundaryData = boundaryGroup.create_dataset('values', (parallelSize[0], 3), 'S100') 
        boundaryData[parallelStart[0]:parallelEnd[0]] = boundary

        fieldGroup = boundaryGroup.create_group('fields')
        index = 1
        for patchID, key, value in boundaryField:
            fieldData = fieldGroup.create_dataset('{}__{}'.format(patchID, key), (parallelSize[index], value.shape[1]), np.float64)
            fieldData[parallelStart[index]:parallelEnd[index]] = value
            index += 1

    def buildBeforeWrite(self):
        self.populateSizes()
        # mesh computation
        # uses neighbour
        self.cellFaces = self.getCellFaces()     # nInternalCells
        # time consuming 
        if config.device == 'cpu':
            self.cells = getCells(self)

    def populateSizes(self):
        self.nInternalFaces = len(self.neighbour)
        self.nFaces = len(self.owner)
        self.nBoundaryFaces = self.nFaces-self.nInternalFaces
        self.nInternalCells = np.max(self.owner)+1
        self.nGhostCells = self.nBoundaryFaces
        self.nCells = self.nInternalCells + self.nGhostCells

    # start: need to convert to ad
    def getCellFaces(self):
        logger.info('generated cell faces') 
        enum = lambda x: np.column_stack((np.indices(x.shape, np.int32)[0], x)) 
        combined = np.concatenate((enum(self.owner), enum(self.neighbour)))
        cellFaces = combined[combined[:,1].argsort(), 0]
        # todo: make it a list ( investigate np.diff )
        cellFaces = cellFaces.reshape(self.nInternalCells, len(cellFaces)/self.nInternalCells)
        return cellFaces

    def getCellNeighbours(self, boundary=True):
        neighbour = self.neighbour.copy()
        if not boundary:
            neighbour[self.nInternalFaces:] = -1
        cellNeighbours = self.owner[self.cellFaces]
        indices = np.arange(0, self.nInternalCells).reshape(-1,1)
        indices = np.equal(cellNeighbours, indices)
        cellNeighbours[indices] = neighbour[self.cellFaces][indices]
        return cellNeighbours

    def getNormals(self):
        logger.info('generated normals')
        # normal goes from owner to neighbour
        v1 = self.points[self.faces[:,1]]-self.points[self.faces[:,2]]
        v2 = self.points[self.faces[:,2]]-self.points[self.faces[:,3]]
        # CROSS product makes it F_CONTIGUOUS even if normals is not
        normals = np.cross(v1, v2)
        # change back to contiguous
        normals = np.ascontiguousarray(normals)
        return normals / norm(normals, axis=1, keepdims=True)

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
        sumVolumes = np.ascontiguousarray(sumVolumes)
        return cellCentres, sumVolumes

    def getDeltas(self):
        logger.info('generated deltas')
        P = self.cellCentres[self.owner]
        N = self.cellCentres[self.neighbour]
        deltas = P-N
        deltasNorm = norm(deltas, axis=1, keepdims=True)
        deltasUnit = deltas/deltasNorm
        return deltasNorm, deltasUnit

    def getWeights(self):
        logger.info('generated face deltas')
        P = self.cellCentres[self.owner]
        N = self.cellCentres[self.neighbour]
        F = self.faceCentres
        neighbourDist = np.abs(np.sum((F-N)*self.normals, axis=1, keepdims=True))
        ownerDist = np.abs(np.sum((F-P)*self.normals, axis=1, keepdims=True))
        weights = neighbourDist/(neighbourDist + ownerDist)

        # weights generally for all faces
        # for actual boundary faces (no coupled) does not matter, NOT USED
        combinedLinearWeights = np.zeros((self.nFaces, 2), config.precision)
        combinedQuadraticWeights = np.zeros((self.nFaces, 2, 3), config.precision)
        index = 0
        for C, D in [[self.owner, self.neighbour], [self.neighbour, self.owner]]:
            R = self.cellCentres[D] - self.cellCentres[C]
            F = self.faceCentres - self.cellCentres[C]
            w = (F*R).sum(axis=1)/(R*R).sum(axis=1)

            # central scheme
            #linearWeights = w
            #quadraticWeights = 0.*F
            # upwind biased: gradient based
            #linearWeights = 0.
            #quadraticWeights = F
            # upwind biased: gradient + central difference
            linearWeights = 1./3*w
            quadraticWeights = 2./3*F + 1./3*(F-w.reshape(-1,1)*R)
            combinedLinearWeights[:, index] = linearWeights
            combinedQuadraticWeights[:,index,:] = quadraticWeights
            index += 1

        return weights, combinedLinearWeights, combinedQuadraticWeights

    # end: need to convert to ad

    def checkWeights(self):
        return
        faceOptions = [[self.owner, self.neighbour], [self.neighbour[:self.nInternalFaces], self.owner[:self.nInternalFaces]]]
        for index in [0, 1]:
            C, D = faceOptions[index] 
            end = len(C)
            diagonal = self.gradOpDiagonal[C]
            diag_q = (1-self.linearWeights[:end,index]) + (self.quadraticWeights[:end,:,index]*diagonal).sum(axis=1)
            off_diagonal = self.gradOpOffDiagonal[C]
            sum_abs_coeff_q = (self.quadraticWeights[:,:,index][self.cellFaces[C]]*off_diagonal).sum(axis=2)
            indices = np.equal(self.cellNeighbours[C], D.reshape(-1,1))
            sum_abs_coeff_q[indices] += self.linearWeights[:end, index]
            sum_abs_coeff_q = np.abs(sum_abs_coeff_q).sum(axis=1)
            pprint('diag sum ratio:', parallel.min(diag_q/sum_abs_coeff_q))

        return

    def getSumOp(self, mesh):
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
        sumOp = (owner + neighbour).tocoo()
    
        #return adsparse.CSR(sumOp.data, sumOp.indices, sumOp.indptr, sumOp.shape)
        return sumOp

    def getGradOp(self, mesh):
        logger.info('generated grad op')
        cells = np.arange(0, mesh.nInternalCells).reshape(-1,1)
        vol = mesh.volumes
        indices = (mesh.owner[mesh.cellFaces] == cells)
        pos = mesh.weights*mesh.normals*mesh.areas
        neg = mesh.normals*mesh.areas-pos
        diagonal = np.zeros((mesh.nInternalCells, 3), config.precision) 
        row = np.zeros_like(mesh.cellFaces)
        row[:,:] = cells
        column = mesh.cellFaces.copy()
        data = np.zeros(mesh.cellFaces.shape + (3,), config.precision)

        faces = mesh.cellFaces[indices]
        np.add.at(diagonal, mesh.owner[faces], pos[faces])
        column[indices] = mesh.neighbour[faces]
        data[indices] = neg[faces]

        indices = np.logical_not(indices)
        faces = mesh.cellFaces[indices]
        np.add.at(diagonal, mesh.neighbour[faces], -neg[faces])
        column[indices] = mesh.owner[faces]
        data[indices] = -pos[faces]

        self.gradOpDiagonal = diagonal
        self.gradOpOffDiagonal = data
        data = np.vstack((data.reshape(np.prod(data.shape[:-1]), 3), diagonal))
        row = np.concatenate((row.flatten(), cells.flatten()))
        column = np.concatenate((column.flatten(), cells.flatten()))
        data /= vol[row]
        row = (row*3).reshape(-1,1)
        row = np.hstack((row, row+1, row+2)).flatten()
        column = np.repeat(column, 3)
        data = data.flatten()

        shape = (3*mesh.nInternalCells, mesh.nCells)
        gradOp = sparse.coo_matrix((data, (row, column)), shape=shape)

        return gradOp

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

    def getPatchFaceRange(self, patchID):
        patch = self.boundary[patchID]
        startFace = patch['startFace']
        nFaces = patch['nFaces']
        endFace = startFace + nFaces
        return startFace, endFace, nFaces

    def getPatchCellRange(self, patchID, faceRange=None):
        if faceRange:
            startFace, endFace, nFaces = faceRange
        else:
            startFace, endFace, nFaces = self.getPatchFaceRange(patchID)
        delta = self.nInternalCells - self.nInternalFaces
        cellStartFace = startFace + delta
        cellEndFace = endFace + delta
        return cellStartFace, cellEndFace, nFaces

    def getPatchFaceCellRange(self, patchID):
        startFace, endFace, nFaces = self.getPatchFaceRange(patchID)
        cellStartFace, cellEndFace, _ = self.getPatchCellRange(patchID, (startFace, endFace, nFaces))
        return startFace, endFace, cellStartFace, cellEndFace, nFaces

    def getProcessorPatchInfo(self, patchID):
        patch = self.boundary[patchID]
        local = patch['myProcNo']
        remote = patch['neighbProcNo']
        tag = 0
        if patch['type'] == 'processorCyclic':
            commonPatch = patch['referPatch']
            if local > remote:
                commonPatch = self.boundary[commonPatch]['neighbourPatch']
            tag = 1 + self.sortedPatches.index(commonPatch)
        return local, remote, tag

    def createGhostCells(self):
        logger.info('generated ghost cells')
        self.neighbour = np.concatenate((self.neighbour, np.zeros(self.nBoundaryFaces, np.int32)))
        self.cellCentres = np.concatenate((self.cellCentres, np.zeros((self.nBoundaryFaces, 3), config.precision)))
        nLocalCells = self.nInternalCells
        exchanger = Exchanger()
        for patchID in self.boundary:
            patch = self.boundary[patchID]
            startFace, endFace, cellStartFace, cellEndFace, nFaces = self.getPatchFaceCellRange(patchID)
            # empty patches
            if nFaces == 0:
                continue
            elif patch['type'] not in config.processorPatches:
                nLocalCells += nFaces
            # append neighbour
            self.neighbour[startFace:endFace] = range(cellStartFace, cellEndFace)
            if patch['type'] in config.cyclicPatches:
                #print patchID, self.cellCentres[self.owner[startFace:endFace]][0]
                neighbourPatch = self.boundary[patch['neighbourPatch']]   
                neighbourStartFace = neighbourPatch['startFace']
                neighbourEndFace = neighbourStartFace + nFaces
                # apply transformation: single value
                # append cell centres
                patch['loc_neighbourIndices'] = self.owner[neighbourStartFace:neighbourEndFace]
                patch['loc_transform'] = self.faceCentres[startFace]-self.faceCentres[neighbourStartFace]
                self.cellCentres[cellStartFace:cellEndFace] = patch['loc_transform'] + self.cellCentres[patch['loc_neighbourIndices']]
            elif patch['type'] == 'processor':
                patch['neighbProcNo'] = int(patch['neighbProcNo'])
                patch['myProcNo'] = int(patch['myProcNo'])
                local, remote, tag = self.getProcessorPatchInfo(patchID)
                # exchange data
                exchanger.exchange(remote, self.cellCentres[self.owner[startFace:endFace]], self.cellCentres[cellStartFace:cellEndFace], tag)
                tag += self.nLocalPatches + 1
                patch['loc_neighbourIndices'] = self.neighbour[startFace:endFace].copy()
                exchanger.exchange(remote, self.owner[startFace:endFace], patch['loc_neighbourIndices'], tag)
            elif patch['type'] == 'processorCyclic':
                patch['neighbProcNo'] = int(patch['neighbProcNo'])
                patch['myProcNo'] = int(patch['myProcNo'])
                local, remote, tag = self.getProcessorPatchInfo(patchID)
                # apply transformation
                exchanger.exchange(remote, -self.faceCentres[startFace:endFace] + self.cellCentres[self.owner[startFace:endFace]], self.cellCentres[cellStartFace:cellEndFace], tag)
                tag += self.nLocalPatches + 1
                patch['loc_neighbourIndices'] = self.neighbour[startFace:endFace].copy()
                exchanger.exchange(remote, self.owner[startFace:endFace], patch['loc_neighbourIndices'], tag)
            else:
                # append cell centres
                self.cellCentres[cellStartFace:cellEndFace] = self.faceCentres[startFace:endFace]
        exchanger.wait()
        for patchID in self.boundary:
            patch = self.boundary[patchID]
            nFaces = patch['nFaces']
            # empty patches
            if nFaces == 0:
                continue
            startFace, endFace, cellStartFace, cellEndFace, _ = self.getPatchFaceCellRange(patchID)
            if patch['type'] == 'processorCyclic':
                self.cellCentres[cellStartFace:cellEndFace] += self.faceCentres[startFace:endFace]

        return nLocalCells

    def getPointsPerturbation(self, pointsPerturbation):
        mesh = self.origMesh
        meshData = self.points + pointsPerturbation, self.faces, mesh.owner, mesh.neighbour[:mesh.nInternalFaces], \
                self.addressing, copy.deepcopy(mesh.boundary)
        mesh = Mesh()
        mesh.build(meshData, 'constant')
        diff = [getattr(mesh.origMesh, field) - getattr(self.origMesh, field) for field in Mesh.gradFields]
        return diff

    def getPerturbation(self, caseDir=None):
        if caseDir is None:
            caseDir = self.caseDir
        mesh = Mesh.create(caseDir)
        diff = [getattr(mesh.origMesh, field) - getattr(self.origMesh, field) for field in Mesh.gradFields]
        return diff
        
    def makeTensor(self):
        logger.info('making tensor variables')
        for attr in Mesh.gradFields:
            value = getattr(self.parent, attr)
            setattr(self, attr, Tensor(value.shape[1:]))
        for attr in Mesh.intFields:
            value = getattr(self.parent, attr)
            setattr(self, attr, Tensor((1,), scalars=[IntegerScalar()]))

        for attr in Mesh.constants:
            value = getattr(self.parent, attr)
            setattr(self, attr, IntegerScalar())

        #for attr in Mesh.constants:
        #    setattr(self, attr, ad.placeholder(ad.int32))
        #for attr in Mesh.fields:
        #    value = getattr(self, attr) 
        #    if attr in ['owner', 'neighbour']:
        #        setattr(self, attr, ad.placeholder(ad.int32))
        #    elif attr in ['cellNeighbours', 'cellFaces']:
        #        setattr(self, attr, ad.placeholder(ad.int32))
        #    elif attr == 'sumOp' or attr == 'gradOp':
        #        setattr(self, attr, self.getSparseTensor(attr))
        #    elif value.shape[1] == 1:
        #        setattr(self, attr, ad.placeholder(config.dtype))
        #    elif len(value.shape) > 2:
        #        setattr(self, attr, ad.placeholder(config.dtype))
        #    else:
        #        setattr(self, attr, ad.placeholder(config.dtype))

        #for patchID in self.localPatches:
        #    for attr in self.getBoundaryTensor(patchID):
        #        self.boundary[patchID][attr[0]] = attr[1]

    def getSparseTensor(self, attr):
        indices = ad.placeholder(ad.int64)
        values = ad.placeholder(config.dtype)
        shape = ad.placeholder(ad.int64)
        sparse = getattr(self.origMesh, attr)
        sparse.indices = np.stack((sparse.row, sparse.col), axis=1).astype(np.int64)
        sparse.values = sparse.data
        sparse.dense_shape = np.array(sparse.shape).astype(np.int64)
        sumOp = ad.SparseTensor(indices, values, shape)
        return sumOp


    def getBoundaryTensor(self, patchID):
        default = [('startFace', ad.placeholder(ad.int32)), ('nFaces', ad.placeholder(ad.int32))]
        return default + self.boundaryTensor.get(patchID, [])

    @config.timeFunction('Time to update mesh')
    def update(self, t, dt):
        logger.info('updating mesh')
        for patchID in self.localPatches:
            if self.boundary[patchID]['type'] == 'slidingPeriodic1D':
                self.updateSlidingPatch(patchID, t, dt)

    def initSlidingPatch(self, patchID):
        mesh = self.origMesh
        patch = mesh.boundary[patchID]
        nFaces = patch['nFaces']

        if nFaces == 0:
            self.boundaryTensor[patchID] = [('loc_multiplier', adsparse.csr_matrix(dtype=config.dtype))]
            return

        patch['nLayers'] = int(patch['nLayers'])
        patch['nFacesPerLayer'] = nFaces/patch['nLayers']
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
        if np.linalg.norm(patch['loc_fixedCellCentres'][index1] + patch2['loc_transform'] - patch['loc_fixedCellCentres'][index2]) > np.max(dists):
            index2, index1 = index1, index2
        point1 = patch['loc_fixedCellCentres'][index2] + patch1['loc_transform']
        point2 = patch['loc_fixedCellCentres'][index1] + patch2['loc_transform']
        #print patchID, mesh.cellCentres[mesh.owner[startFace:endFace]][0], point1, patch['velocity']
        # reread
        if 'movingCellCentres' in patch:
            patch['movingCellCentres'] = extractField(patch['movingCellCentres'], patch['nFacesPerLayer']+1, (3,))
        else:
            patch['movingCellCentres'] = np.vstack((patch['loc_fixedCellCentres'].copy(), point2))
        patch['loc_periodicLimit'] = point1
        patch['loc_extraIndex'] = index1
        self.boundaryTensor[patchID] = [('loc_multiplier', adsparse.csr_matrix(dtype=config.dtype))]
        return

    def updateSlidingPatch(self, patchID, t, dt):
        mesh = self.origMesh
        patch = mesh.boundary[patchID]
        nFaces = patch['nFaces']

        # single processor has everything
        if dt == 0.:
            self.initSlidingPatch(patchID)

        if nFaces == 0:
            patch['movingCellCentres'] = np.zeros((0, 3), config.precision)
            patch['loc_multiplier'] = sparse.csr_matrix((0,0), dtype=config.precision)
            patch['loc_velocity'] = np.fromstring(patch['velocity'][1:-1], sep=' ', dtype=config.precision)
            return

        mesh = self.origMesh
        patch['movingCellCentres'] += patch['loc_velocity']*dt
        # only supports low enough velocities
        transformIndices = (patch['movingCellCentres']-patch['loc_periodicLimit']).dot(patch['loc_velocity']) > 1e-6
        #print patchID, patch['movingCellCentres'][0], patch['loc_periodicLimit'], patch['loc_velocity']
        patch['movingCellCentres'][transformIndices] += mesh.boundary[mesh.boundary[patch['periodicPatch']]['neighbourPatch']]['loc_transform']
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
        loc_multiplier = sparse.coo_matrix((weights, (indices, sortedDists)), shape=(nFaces, nFaces)).tocsr()
        if 'loc_multiplier' not in patch:
            patch['loc_multiplier'] = loc_multiplier
        else:
            patch['loc_multiplier'].data = loc_multiplier.data
            patch['loc_multiplier'].indices = loc_multiplier.indices
            patch['loc_multiplier'].indptr = loc_multiplier.indptr
        #print patch['movingCellCentres']
        #print 'set', id(patch['loc_multiplier'].data)
        return

    @config.timeFunction('Time to decompose mesh')
    def decompose(self, nprocs):
        assert parallel.nProcessors == 1
        start = time.time()
        pprint('decomposing mesh to', nprocs, 'processors')
        decomposed, addressing = decompose(self, nprocs)
        for n in range(0, nprocs):
            pprint('writing processor{}'.format(n))
            points, faces, owner, neighbour, boundary = decomposed[n]
            pointProcAddressing, faceProcAddressing, cellProcAddressing, boundaryProcAddressing = addressing[n]
            meshCase = self.case + 'processor{}/constant/polyMesh/'.format(n)
            if not os.path.exists(meshCase):
                os.makedirs(meshCase)
            self.writeFoamFile(meshCase + 'points', points)
            self.writeFoamFile(meshCase + 'faces', faces)
            self.writeFoamFile(meshCase + 'owner', owner)
            self.writeFoamFile(meshCase + 'neighbour', neighbour)
            self.writeFoamBoundary(meshCase + 'boundary', boundary)

            self.writeFoamFile(meshCase + 'pointProcAddressing', pointProcAddressing)
            self.writeFoamFile(meshCase + 'faceProcAddressing', faceProcAddressing)
            self.writeFoamFile(meshCase + 'cellProcAddressing', cellProcAddressing)
            self.writeFoamFile(meshCase + 'boundaryProcAddressing', boundaryProcAddressing)

        pprint('Time for decomposing mesh:', time.time()-start)
        pprint()
        return decomposed, addressing

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
            internalField = np.zeros((size, ) + dimensions, dtype=np.float64)
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
                handle.write('(' + ' '.join(np.char.mod('%.30f', value)) + ')\n')
    handle.write(')\n;\n')



