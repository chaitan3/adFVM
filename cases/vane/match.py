#!/usr/bin/env python
from matplotlib import pyplot as plt, mlab
from matplotlib import markers as mk
import os
import sys
from profile import get_length, pressure, suction, c, pitch
from numpy import *
import csv
import scipy.interpolate as inter
currdir = os.path.dirname(os.path.realpath(__file__))


def read_data(name):
    with open(currdir + '/' + name) as f:
        expe = array(list(csv.reader(f)))[:,:-1].astype(float)
    return expe

def match_htc(hp, coordsp, hs, coordss, saveFile):

    sp = -get_length(pressure, coordsp)*1000
    ss = get_length(suction, coordss)*1000

    expe = read_data('data/htc_1.csv')

    fill = 1

    plt.scatter(expe[:,0], expe[:,1], c='k', alpha=fill,marker='o', label='Experiment')
    plt.scatter(sp, hp, c='r', s=10, alpha=fill,marker='+', label='Simulation')
    plt.scatter(ss, hs, c='r', s=10, alpha=fill, marker='+')
    plt.xlabel('s/c (mm)')
    plt.ylabel('HTC (W/m2K)')

    plt.savefig(saveFile)
    plt.clf()


def match_velocity(Map, coordsp, Mas, coordss, saveFile):

    sp = get_length(pressure, coordsp)/c
    ss = get_length(suction, coordss)/c

    expp = read_data('data/Ma_pressure_0.875.csv')
    exps = read_data('data/Ma_suction_0.875.csv')

    fill=1

    plt.scatter(expp[:,0], expp[:,1], c='r', marker='+', label='Exp. pressure')
    plt.scatter(exps[:,0], exps[:,1], c='b', marker='+', label='Exp. suction')
    plt.scatter(sp, Map, c='r', s=10, alpha=fill, marker='o', label='Sim. pressure')
    plt.scatter(ss, Mas, c='b', s=10, alpha=fill, marker='o', label='Sim. suction')
    plt.xlim([0, ss.max()])
    plt.ylim([0, Mas.max()])
    plt.xlabel('s/c (mm)')
    plt.ylabel('Ma')
    plt.legend(loc='lower right')

    plt.savefig(saveFile)
    plt.clf()

def match_wakes(pl, coords, p0, saveFile):

    p0 = p0*0.0075
    expe = read_data('data/wake_0.85.csv')
    expe[:,1] = expe[:,1]/p0
    plt.scatter(expe[:,0], expe[:,1], c='r', label='Experimental')

    #coords = (coords-coords.min())/pitch

    #y = inter.UnivariateSpline(coords, pl, s=0.01)
    #x = linspace(coords.min(), coords.max(), 100)
    #plt.plot(x, y(x), c='k', label='Simulation fit')
    
    plt.scatter(coords, pl, c='b', label='Simulation')
    plt.xlabel('y (mm)')
    plt.ylabel('pressure loss coeff')
    plt.legend()

    plt.savefig(saveFile)
    plt.clf()


if __name__ == '__main__':
    from adFVM import config
    from adFVM.mesh import Mesh
    from adFVM.field import Field, IOField
    config.hdf5 = True

    case, time = sys.argv[1:3]
    time = float(time)
    mesh = Mesh.create(case)
    Field.setMesh(mesh)

    with IOField.handle(time):
        htc = IOField.read('htc_avg')
        htc.partialComplete()
        Ma = IOField.read('Ma_avg')
        Ma.partialComplete()
        join = '/'
        if config.hdf5:
            join = '_'
        with open(mesh.getTimeDir(time) + join + 'wake_avg', 'r') as f:
            data = load(f)
            wakeCells, pl = data['arr_0'], data['arr_1']

    #nLayers = 1
    nLayers = 200
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

        spanwise_average = lambda x: x.reshape((nLayers, nFacesPerLayer)).sum(axis=0)/nLayers
        y = spanwise_average(htc.field[cellStartFace:cellEndFace])
        htc_args.extend([y, x])
        y = spanwise_average(Ma.field[cellStartFace:cellEndFace])
        Ma_args.extend([y, x])
    htc_args += [case + 'htc.png']
    Ma_args += [case + 'Ma.png']
    match_velocity(*Ma_args)
    match_htc(*htc_args)

    p0 = 175158.
    nCellsPerLayer = len(wakeCells)/nLayers
    y = pl[:nCellsPerLayer]/p0
    x = 1000*mesh.cellCentres[wakeCells[:nCellsPerLayer], 1]
    wake_args = [y, x, p0, case + 'wake.png']
    match_wakes(*wake_args)

