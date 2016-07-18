#!/usr/bin/python2
from adFVM import config
from adFVM.mesh import Mesh
from adFVM.field import Field, IOField

import os, shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('case')
parser.add_argument('time', type=float)
user = parser.parse_args(config.args)

mesh = Mesh.create(user.case)
Field.setMesh(mesh)
timeDir = mesh.getTimeDir(user.time)
for boundaryDir in [mesh.case + '/constant/polyMesh/']:#, timeDir + 'polyMesh/']:
    if os.path.exists(boundaryDir + 'boundary.sliding'):
        shutil.copyfile(boundaryDir + 'boundary.sliding', boundaryFile)
        continue
    boundaryFile = boundaryDir + 'boundary'
    shutil.copyfile(boundaryFile, boundaryDir + 'boundary.cyclic')
    # intersection_master on the right, intersection_slave on the left (x-axis)
    patch = mesh.origMesh.boundary['intersection_master']
    patch['type'] = 'slidingPeriodic1D'
    patch['periodicPatch'] = 'mid1plane'
    patch['velocity'] = '(0 252 0)'
    #patch['nLayers'] = '1'
    patch['nLayers'] = '10'
    patch.pop('movingCellCentres', None)
    patch = mesh.origMesh.boundary['intersection_slave']
    patch['type'] = 'slidingPeriodic1D'
    patch['periodicPatch'] = 'mid2plane'
    patch['velocity'] = '(0 -252 0)'
    #patch['nLayers'] = '1'
    patch['nLayers'] = '10'
    patch.pop('movingCellCentres', None)
    mesh.writeFoamBoundary(boundaryFile, mesh.origMesh.boundary)

fields = os.listdir(timeDir)
for phi in fields:
    if phi == 'polyMesh':
        continue
    with IOField.handle(user.time):
        field = IOField.read(phi)
        field.boundary['intersection_master']['type'] = 'slidingPeriodic1D'
        field.boundary['intersection_slave']['type'] = 'slidingPeriodic1D'
        field.write(skipProcessor=True)
