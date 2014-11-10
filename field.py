
import numpy as np
from os import makedirs
from os.path import exists
from numbers import Number
import re


from config import ad, Logger
from parallel import pprint, Exchanger
logger = Logger(__name__)
import config, parallel
import BCs
from mesh import extractField, writeField
#import pdb; pdb.set_trace()


class Field(object):
    @staticmethod
    def setSolver(solver):
        Field.solver = solver
        Field.mesh = solver.mesh
    @staticmethod
    def setMesh(mesh):
        Field.mesh = mesh

    def __init__(self, name, field):
        self.name = name
        self.field = field
        self.size = field.shape[0]
        self.dimensions = field.shape[1:]

    @classmethod
    def max(self, a, b):
        a_gt_b = ad.value(a.field) > ad.value(b.field)
        b_gt_a = 1 - a_gt_b
        return self('max({0},{1})'.format(a.name, b.name), a.field * ad.array(a_gt_b) + b.field * ad.array(b_gt_a))

    def info(self):
        pprint(self.name + ':', end='')
        fieldMin = parallel.min(ad.value(self.field))
        fieldMax = parallel.max(ad.value(self.field))
        assert not np.isnan(fieldMin)
        assert not np.isnan(fieldMax)
        pprint(' min:', fieldMin, 'max:', fieldMax)

    # creates a view
    def component(self, component): 
        assert self.dimensions == (3,)
        return self.__class__('{0}.{1}'.format(self.name, component), self.field[:, component].reshape((-1,1)))

    def magSqr(self):
        assert self.dimensions == (3,)
        return self.__class__('magSqr({0})'.format(self.name), ad.sum(self.field**2, axis=1).reshape((-1,1)))

    def mag(self):
        return self.magSqr()**0.5

    # modifies in place
    def stabilise(self, num):
        pos =  ad.value(self.field) > 0.
        neg = 1 - pos
        return self.__class__('abs({0})'.format(self.name), (self.field + num)*pos + (self.field - num)*neg)

    def abs(self):
        return self.__class__('abs({0})'.format(self.name), self.field * ad.array(2*((ad.value(self.field) > 0) - 0.5)))

    def dot(self, phi):
        assert self.dimensions[0] == 3
        # if tensor
        if len(self.dimensions) > 1:
            phi = self.__class__(phi.name, phi.field[:,np.newaxis,:])
        product = ad.sum(self.field * phi.field, axis=-1)
        # if summed over vector
        if len(product.shape) == 1:
            product = product.reshape((-1,1))
        return self.__class__('dot({0},{1})'.format(self.name, phi.name), product)

    def dotN(self):
        return self.dot(self.mesh.Normals)

    def outer(self, phi):
        return self.__class__('outer({0},{1})'.format(self.name, phi.name), self.field[:,:,np.newaxis] * phi.field[:,np.newaxis,:])
    
    # creates a view
    def transpose(self):
        assert len(self.dimensions) == 2
        return self.__class__('{0}.T'.format(self.name), self.field.transpose((0,2,1)))

    def trace(self):
        assert len(self.dimensions) == 2
        phi = self.field
        return self.__class__('tr({0})'.format(self.name), (phi[:,0,0] + phi[:,1,1] + phi[:,2,2]).reshape((-1,1)))

    def __neg__(self):
        return self.__class__('-{0}'.format(self.name), -self.field)

    def __mul__(self, phi):
        if isinstance(phi, Number):
            return self.__class__('{0}*{1}'.format(self.name, phi), self.field * phi)
        else:
            product = self.field * phi.field
            return self.__class__('{0}*{1}'.format(self.name, phi.name), self.field * phi.field)


    def __rmul__(self, phi):
        return self * phi

    def __pow__(self, power):
        return self.__class__('{0}**{1}'.format(self.name, power), self.field.__pow__(power))

    def __add__(self, phi):
        if isinstance(phi, Number):
            return self.__class__('{0}+{1}'.format(self.name, phi), self.field + phi)
        else:
            return self.__class__('{0}+{1}'.format(self.name, phi.name), self.field + phi.field)

    def __radd__(self, phi):
        return self.__add__(phi)

    def __sub__(self, phi):
        return self.__add__(-phi)

    def __div__(self, phi):
        return self.__class__('{0}/{1}'.format(self.name, phi.name), self.field / phi.field)

class CellField(Field):
    def __init__(self, name, field, boundary={}):
        logger.debug('initializing field {0}'.format(name))
        super(self.__class__, self).__init__(name, field)
        mesh = self.mesh

        if not hasattr(mesh, 'Normals'):
            mesh.Normals = Field('nF', ad.array(mesh.normals))

        if len(list(boundary.keys())) == 0:
            self.boundary = mesh.defaultBoundary
        else:
            self.boundary = boundary

        if self.size == mesh.nInternalCells:
            self.field = ad.zeros((mesh.nCells,) + self.dimensions)

        self.BC = {}
        for patchID in self.boundary:
            # skip empty patches
            if mesh.boundary[patchID]['nFaces'] == 0:
                continue
            self.BC[patchID] = getattr(BCs, self.boundary[patchID]['type'])(self, patchID)

        if self.size == mesh.nInternalCells:
            self.setInternalField(field)
            self.size = self.field.shape[0]


    @classmethod
    def zeros(self, name, dimensions):
        logger.info('initializing zeros field {0}'.format(name))
        return self(name, ad.zeros((self.mesh.nCells,) + dimensions))

    @classmethod
    def copy(self, phi):
        logger.info('copying field {0}'.format(phi.name))
        return self(phi.name, ad.array(ad.value(phi.field).copy()), phi.boundary.copy())

    @classmethod
    def read(self, name, time):
        mesh = self.mesh
        if time.is_integer():
            time = int(time)
        pprint('reading field {0}, time {1}'.format(name, time))
        timeDir = '{0}/{1}/'.format(mesh.case, time)

        content = open(timeDir + name).read()
        foamFile = re.search(re.compile('FoamFile\n{(.*?)}\n', re.DOTALL), content).group(1)
        assert re.search('format[\s\t]+(.*?);', foamFile).group(1) == config.fileFormat
        vector = re.search('class[\s\t]+(.*?);', foamFile).group(1) == 'volVectorField'
        bytesPerField = 8*(1 + 2*vector)
        startBoundary = content.find('boundaryField')
        data = re.search(re.compile('internalField[\s\r\n]+(.*)', re.DOTALL), content[:startBoundary]).group(1)
        internalField = extractField(data, mesh.nInternalCells, vector)
        content = content[startBoundary:]
        boundary = {}
        def getToken(x): 
            token = re.match('[\s\r\n\t]+([a-zA-Z0-9_\.\-\+<>\{\}]+)', x)
            return token.group(1), token.end()
        for patchID in mesh.boundary:
            patch = re.search(re.compile('[\s\r\n\t]+' + patchID + '[\s\r\n]+{', re.DOTALL), content)
            boundary[patchID] = {}
            start = patch.end()
            while 1:
                key, end = getToken(content[start:])
                start += end
                if key == '}':
                    break
                # skip non binary, non value, uniform or empty patches
                elif key == 'value' and config.fileFormat == 'binary' and getToken(content[start:])[0] != 'uniform' and mesh.boundary[patchID]['nFaces'] != 0:
                    match = re.search(re.compile('[ ]+(nonuniform[ ]+List<[a-z]+>[\s\r\n\t0-9]*\()', re.DOTALL), content[start:])
                    nBytes = bytesPerField * mesh.boundary[patchID]['nFaces']
                    start += match.end()
                    prefix = match.group(1)
                    boundary[patchID][key] = prefix + content[start:start+nBytes]
                    start += nBytes
                    match = re.search('\)[\s\r\n\t]*;', content[start:])
                    start += match.end()
                else:
                    match = re.search(re.compile('[ ]+(.*?);', re.DOTALL), content[start:])
                    start += match.end() 
                    boundary[patchID][key] = match.group(1)

        return self(name, internalField, boundary)

    def write(self, time):
        if time.is_integer():
            time = int(time)
        assert len(self.dimensions) == 1
        np.set_printoptions(precision=16)
        pprint('writing field {0}, time {1}'.format(self.name, time))
        timeDir = '{0}/{1}/'.format(self.mesh.case, time)
        if not exists(timeDir):
            makedirs(timeDir)
        handle = open(timeDir + self.name, 'w')
        handle.write(config.foamHeader)
        handle.write('FoamFile\n{\n')
        foamFile = config.foamFile.copy()
        foamFile['object'] = self.name
        if self.dimensions[0] == 3:
            dtype = 'vector'
            foamFile['class'] = 'volVectorField'
        else:
            dtype = 'scalar'
            foamFile['class'] = 'volScalarField'
        for key in foamFile:
            handle.write('\t' + key + ' ' + foamFile[key] + ';\n')
        handle.write('}\n')
        handle.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
        handle.write('dimensions      [0 1 -1 0 0 0 0];\n')
        writeField(handle, ad.value(self.getInternalField()), dtype, 'internalField')
        handle.write('boundaryField\n{\n')
        for patchID in self.boundary:
            handle.write('\t' + patchID + '\n\t{\n')
            patch = self.boundary[patchID]
            for attr in patch:
                handle.write('\t\t' + attr + ' ' + patch[attr] + ';\n')
            if patch['type'] in config.valuePatches:
                writeField(handle, self.BC[patchID].getValue(), dtype, 'value')
            handle.write('\t}\n')
        handle.write('}\n')
        handle.close()

    def setInternalField(self, internalField):
        self.field[:self.mesh.nInternalCells] = internalField
        self.updateGhostCells()

    def getInternalField(self):
        return self.field[:self.mesh.nInternalCells]

    def updateGhostCells(self):
        logger.info('updating ghost cells for {0}'.format(self.name))
        exchanger = Exchanger()
        for patchID in self.BC:
            if self.boundary[patchID]['type'] in config.processorPatches:
                self.BC[patchID].update(exchanger)
            else:
                self.BC[patchID].update()
        exchanger.wait()


