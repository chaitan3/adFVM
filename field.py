import numpy as np
from os import makedirs
from os.path import exists

def write(field, timeDir, fieldFile):
    if not exists(timeDir):
        makedirs(timeDir)
    handle = open(timeDir + fieldFile, 'w')
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
    handle.write('{0}\n(\n'.format(len(field)))
    np.savetxt(handle, field)
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


