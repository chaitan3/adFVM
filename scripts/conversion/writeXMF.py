#!/usr/bin/python2
import os
import numpy as np
import sys
import h5py
import re

from adFVM import config

case = sys.argv[1]
name = os.path.basename(case.rstrip('/'))

serial = ''
#serial = '_serial'
offset = 5 + len(serial)

xmfFile = case + name + serial + '.xmf'
print 'writing xmf ' +  xmfFile

xmf = open(xmfFile, 'w')
xmf.write("""<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
<Domain>
""")

meshFile = case + 'mesh{}.hdf5'.format(serial)
timeFiles = [x for x in os.listdir(case) if config.isfloat(x[:-5]) and x.endswith('.hdf5')]
timeFiles = sorted(timeFiles, key=lambda x: float(x[:-5]))
times = [x[:-5] for x in timeFiles]
timeFiles = [case + x for x in timeFiles]

mesh = h5py.File(meshFile, 'a')
parallelStart = mesh['parallel/start']
parallelEnd = mesh['parallel/end']
nProcs = parallelStart.shape[0]
#if 'faces_xmf' not in mesh:
#    mesh['faces_xmf'] = mesh['faces'][:,1:]

xmf.write("""
    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
        <Time TimeType="List">
                <DataItem Format="XML" NumberType="Float" Dimensions="3">
                """ + ' '.join(times) + """
            </DataItem>
        </Time>
""")

def getDataString(field, data, start, end):
    nDims = len(data.shape)
    count = end-start
    totalShape = ' '.join([str(x) for x in data.shape])
    shape = ' '.join([str(x) for x in (count,) + data.shape[1:]])
    dim2 = data.shape[1]

    dtype = None
    precision = None
    if data.dtype == np.float64:
        dtype = 'Float'
        precision = '8'
    elif data.dtype == np.float32:
        dtype = 'Float'
        precision = 4
    elif data.dtype == np.int32:
        dtype = 'Int'
        precision = 4

    return """
    <DataItem ItemType="HyperSlab" Dimensions="{shape}" Type="HyperSlab">
                        <DataItem Dimensions="3 {nDims}" Format="XML">
                            {start} 0
                            1 1
                            {count} {dim2}
                        </DataItem>
                        <DataItem NumberType="{dtype}" Precision="{precision}" Format="HDF" Dimensions="{totalShape}">
                            {field}
                        </DataItem>
                    </DataItem>
    """.format(**locals())

for index, timeFile in enumerate(timeFiles):
    
    time = h5py.File(timeFile, 'r')

    xmf.write("""
        <Grid GridType="Collection" CollectionType="Spatial">
    """)
    for proc in range(0, nProcs):
        nCells = parallelEnd[proc,4]-parallelStart[proc,4]
        meshName = os.path.basename(meshFile)
        topologyString = getDataString(meshName + ':/cells', mesh['cells'], parallelStart[proc,4], parallelEnd[proc,4])
        geometryString = getDataString(meshName + ':/points', mesh['points'], parallelStart[proc,1], parallelEnd[proc,1])
        xmf.write("""
            <Grid Name="{0}">
                <Topology NumberOfElements="{1}" TopologyType="Hexahedron">
                    {2}
                </Topology>
                <Geometry GeometryType="XYZ">
                    {3}
                </Geometry>
        """.format('{}_{}_{}'.format(name, index, proc), nCells, topologyString, geometryString))

        for fieldName in time.keys():
            if fieldName == 'mesh':
                continue
            phi = time[fieldName]['field']

            start = time[fieldName]['parallel/start']
            end = time[fieldName]['parallel/end']
            fieldType = 'Scalar'
            if phi.shape[1] == 3:
                fieldType = 'Vector'
            timeName = os.path.basename(timeFile)
            fieldString = getDataString('{}:/{}/field'.format(timeName, fieldName), phi, start[proc,0], start[proc,0]+ nCells)
            xmf.write("""
                <Attribute Name="{0}" Center="Cell" AttributeType="{1}">
                    {2}
                </Attribute>
            """.format(fieldName, fieldType, fieldString))
        xmf.write("""
            </Grid>         
        """)

    time.close()
    xmf.write("""
        </Grid>
    """)

xmf.write("""
    </Grid>
</Domain>
</Xdmf>
""")

xmf.close()
mesh.close()

