#!/usr/bin/python2
import config
from mesh import Mesh
from field import Field, IOField

import os, shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('case')
parser.add_argument('time', type=float)
user = parser.parse_args(config.args)

mesh = Mesh.create(user.case)
Field.setMesh(mesh)
#mesh.writeBoundary(mesh.meshDir + 'boundary')
shutil.copyfile(mesh.meshDir + 'boundary', mesh.meshDir + 'boundary.cyclic')
patch = mesh.origMesh.boundary['intersection_master']
patch['type'] = 'slidingPeriodic1D'
patch['periodicPatch'] = 'mid2plane'
patch['velocity'] = '(0 252 0)'
patch['nLayers'] = '1'
patch = mesh.origMesh.boundary['intersection_slave']
patch['type'] = 'slidingPeriodic1D'
patch['periodicPatch'] = 'mid1plane'
patch['velocity'] = '(0 -252 0)'
patch['nLayers'] = '1'
mesh.writeBoundary(mesh.meshDir + 'boundary')

timeDir = mesh.getTimeDir(user.time)
fields = os.listdir(timeDir)
for phi in fields:
    field = IOField.read(phi, mesh, user.time)
    field.boundary['intersection_master']['type'] = 'slidingPeriodic1D'
    field.boundary['intersection_slave']['type'] = 'slidingPeriodic1D'
    field.write(user.time, skipProcessor=True)
