import numpy as np
from os import makedirs
from os.path import exists

class Field:
    def __init__(self, name, mesh, field):
        self.name = name
        self.mesh = mesh
        self.field = field
        self.dimensions = field.shape[1]

    @classmethod
    def zeros(self, name, mesh, dimensions):
        return self(name, mesh, np.zeros((mesh.nCells, dimensions)))

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
        np.savetxt(handle, self.field[:self.mesh.nInternalCells])
        handle.write(')\n;\n')
        handle.write('''
boundaryField
{
    patch0_half0
    {
        type cyclic;
    }
    patch0_half1
    {
        type cyclic;
    }
    patch1_half0
    {
        type cyclic;
    }
    patch1_half1
    {
        type cyclic;
    }
    patch2_half0
    {
        type cyclic;
    }
    patch2_half1
    {
        type cyclic;
    }
}
''')
        handle.close()



class FaceField(Field):
    @classmethod
    def zeros(self, name, mesh, dimensions):
        return self(name, mesh, np.zeros((mesh.nFaces, dimensions)))


