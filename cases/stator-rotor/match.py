#!/usr/bin/env python
from matplotlib import pyplot as plt, mlab
from matplotlib import markers as mk
import os

from profile import *
from numpy import *

pressure = suction = chord = pitch = None

def get_profile(name):
    global pressure, suction, chord, pitch
    if name == 'nozzle':
        var = import_nozzle_profile()
    else:
        var = import_blade_profile()
    pressure, suction, chord, pitch = var

def match_htc(hp, coordsp, hs, coordss, saveFile):
    sp = get_length(pressure, coordsp)
    ss = get_length(suction, coordss) 

    fill = 1

    plt.scatter(-sp*1000, hp, c='r', s=10, 
            alpha=fill,marker='+', label='pressure')
    plt.scatter(ss*1000, hs, c='b', s=10, 
            alpha=fill, marker='+', label='suction')
    plt.xlabel('s/c (mm)')
    plt.ylabel('HTC (W/m2 K)')

    plt.legend()
    plt.savefig(saveFile)
    plt.clf()

def match_velocity(Map, coordsp, Mas, coordss, saveFile):
    sp = get_length(pressure, coordsp)
    ss = get_length(suction, coordss)

    fill=1

    plt.scatter(sp/chord, Map, c='r', s=10, alpha=fill, label='pressure')
    plt.scatter(ss/chord, Mas, c='b', s=10, alpha=fill, label='suction')
    plt.xlabel('s/c (mm)')
    plt.ylabel('isentropic Ma')

    plt.legend()
    plt.savefig(saveFile)
    plt.clf()

def match_wakes(pl, coords, saveFile):
    coords = (coords-coords.min())/pitch
    plt.scatter(coords, pl)
    plt.xlabel('y (mm)')
    plt.ylabel('pressure loss coeff')

    plt.savefig(saveFile)
    plt.clf()

if __name__ == '__main__':
    from adFVM.mesh import Mesh
    from adFVM.field import Field, IOField
    import sys

    case, time = sys.argv[1:3]
    time = float(time)
    mesh = Mesh.create(case)
    Field.setMesh(mesh)

    with IOField.handle(time):
        htc = IOField.read('htc_avg')
        htc.partialComplete()
        Ma = IOField.read('Ma_avg')
        Ma.partialComplete()

    nLayers = 1
    #nLayers = 200
    patches = ['pressure', 'suction']
   
    htc_args = []
    Ma_args = []

    mesh = mesh.origMesh
    for patchID in patches:
        delta = -mesh.nInternalFaces + mesh.nInternalCells
        startFace = mesh.boundary[patchID]['startFace']
        nFaces = mesh.boundary[patchID]['nFaces']
        endFace = startFace + nFaces
        cellStartFace = startFace + delta
        cellEndFace = endFace + delta
        nFacesPerLayer = nFaces/nLayers

        x = mesh.faceCentres[startFace:endFace, 0]
        x = x[:nFacesPerLayer]

        y = htc.field[cellStartFace:cellEndFace]
        htc_args.extend([y, x])
        y = Ma.field[cellStartFace:cellEndFace]
        Ma_args.extend([y, x])

    htc_args += [case + 'htc.png']
    Ma_args += [case + 'Ma.png']
    match_velocity(*Ma_args)
    match_htc(*htc_args)

    #y = PL/(p0*nTimes)
    #x = 1000*mesh.cellCentres[wakeCells[:nCellsPerLayer], 1]
    #wake_args = [y, x, p0]
    #match_wakes(*wake_args)

