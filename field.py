from __future__ import print_function
import numpy as np
import numpad as ad
from os import makedirs
from os.path import exists
import re

import BCs
import utils

class Field:
    def __init__(self, name, mesh, field):
        self.name = name
        self.mesh = mesh
        self.field = field
        self.size = field.shape[0]
        self.dimensions = field.shape[1]
        self.boundary = {}
        for patch in mesh.boundary:
            self.boundary[patch] = {}
            if mesh.boundary[patch]['type'] == 'cyclic':
                self.boundary[patch]['type'] = 'cyclic'
            else:
                self.boundary[patch]['type'] = 'zeroGradient'

    @classmethod
    def zeros(self, name, mesh, size, dimensions):
        return self(name, mesh, ad.zeros((size, dimensions)))

    @classmethod
    def copy(self, field):
        return self(field.name, field.mesh, field.field.copy())

    def setInternalField(self, internalField):
        mesh = self.mesh
        self.field[:mesh.nInternalCells] = internalField
        self.updateGhostCells()

    def getInternalField(self):
        mesh = self.mesh
        return self.field[:mesh.nInternalCells]

    def updateGhostCells(self):
        mesh = self.mesh
        for patchID in mesh.boundary:
            patch = mesh.boundary[patchID] 
            if patch['type'] == 'cyclic':
                startFace = patch['startFace']
                nFaces = patch['nFaces']
                endFace = startFace + nFaces
                neighbourPatch = mesh.boundary[patch['neighbourPatch']]   
                neighbourStartFace = neighbourPatch['startFace']
                neighbourEndFace = neighbourStartFace + nFaces
                indices = mesh.nInternalCells + range(startFace, endFace) - mesh.nInternalFaces 
                self.field[indices] = self.field[mesh.owner[neighbourStartFace:neighbourEndFace]]
            else:
                boundaryCondition = self.boundary[patchID]['type']
                # pass info
                getattr(BCs, boundaryCondition)()

    def read(self, time):
        timeDir = '{0}/{1}/'.format(self.mesh.case, time)
        content = utils.removeCruft(open(timeDir + self.name, 'r').read())

        data = re.search(re.compile('internalField[\s\r\na-zA-Z<>]+(.*?);', re.DOTALL), content).group(1)
        start = data.find('(')
        if start >= 0:
            internalField = ad.adarray(re.findall('[0-9\.Ee\-]+', data[start:])).reshape((-1,1))
        else:
            internalField = ad.ones((self.mesh.nInternalCells, self.dimensions))
        self.setInternalField(internalField)

        content = content[content.find('boundaryField'):]
        self.boundary = {}
        for patchID in self.mesh.boundary:
            patch = re.search(re.compile(patchID + '[\s\r\n]+{(.*?)}', re.DOTALL), content).group(1)
            self.boundary[patchID] = dict(re.findall('\n[ \t]+([a-zA-Z]+)[ ]+(.*?);', patch))

    def write(self, time):
        timeDir = '{0}/{1}/'.format(self.mesh.case, time)
        if not exists(timeDir):
            makedirs(timeDir)
        handle = open(timeDir + self.name, 'w')
        handle.write('''
    /*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  2.2.2                                 |
|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      T;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];

internalField   nonuniform List<scalar> 
''')
        handle.write('{0}\n(\n'.format(self.mesh.nInternalCells))
        np.savetxt(handle, ad.value(self.getInternalField()))
        handle.write(')\n;\n')
        handle.write('boundaryField\n{\n')
        for patchID in self.boundary:
            handle.write('\t' + patchID + '\n\t{\n')
            patch = self.boundary[patchID]
            for attr in patch:
                handle.write('\t\t' + attr + ' ' + patch[attr] + ';\n')
            handle.write('\t}\n')
        handle.write('}\n')
        handle.close()


