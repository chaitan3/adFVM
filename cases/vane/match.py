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
import pickle as pkl
from smooth import smooth

#config.hdf5 = True

def read_data(name):
    with open(currdir + '/' + name) as f: 
        expe = array(list(csv.reader(f))).astype(float)
    return expe


def match_htc(hp, coordsp, hs, coordss, saveFile):
    #print hp, hs

    sp, indices = get_length(pressure, coordsp)
    sp = -sp/c
    sp, hp = sp[indices], hp[indices]
    indices = sp > -0.95
    sp, hp = sp[indices], hp[indices]

    ss, indices = get_length(suction, coordss)
    ss = ss/c
    ss, hs = ss[indices], hs[indices]
    indices = ss < 1.25
    ss, hs = ss[indices], hs[indices]
    
    hs = smooth(hs, 5)
    hp = smooth(hp, 5)

    expe = read_data('data/htc_0.9_1e6.csv')
    #expe = read_data('data/htc_1.07_1e6.csv')

    fill = 1
    plt.scatter(expe[:,0]/(c*1000), expe[:,1], c='k', alpha=fill,marker='o', label='Experiment')
    #plt.scatter(sp, hp, c='r', s=10, alpha=fill,marker='+', label='Simulation')
    #plt.scatter(ss, hs, c='b', s=10, alpha=fill, marker='+')
    plt.plot(sp, hp, c='r', label='Simulation')
    plt.plot(ss, hs, c='b')
    plt.xlabel('s/c (mm)')
    plt.ylabel('HTC (W/m2K)')

    plt.savefig(saveFile)
    plt.clf()


def match_velocity(Map, coordsp, Mas, coordss, saveFile):

    sp = get_length(pressure, coordsp)[0]/c
    ss = get_length(suction, coordss)[0]/c
    indices = ss < 1.1
    ss = ss[indices]
    Mas = Mas[indices]

    indices = logical_not(isnan(Mas))
    ss, Mas = ss[indices], Mas[indices]
    indices = logical_not(isnan(Map))
    sp, Map = sp[indices], Map[indices]

    expp = read_data('data/Ma_pressure_0.875.csv')
    exps = read_data('data/Ma_suction_0.875.csv')
    #expp = read_data('data/Ma_pressure_1.02.csv')
    #exps = read_data('data/Ma_suction_1.02.csv')

    fill=1

    plt.scatter(expp[:,0], expp[:,1], marker='o', label='Exp. pressure')
    plt.scatter(exps[:,0], exps[:,1], marker='o', label='Exp. suction')
    plt.scatter(sp, Map, c='r', s=10, alpha=fill, marker='+', label='Sim. pressure')
    plt.scatter(ss, Mas, c='b', s=10, alpha=fill, marker='+', label='Sim. suction')
    plt.xlim([0, ss.max()])
    plt.ylim([0, Mas.max()])
    plt.xlabel('s/c (mm)')
    plt.ylabel('Ma')
    #plt.legend(loc='lower right')

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


pklFile = 'match_args.pkl'
if __name__ == '__main__':
    if 0:#os.path.exists(pklFile):
        htc_args, Ma_args = pkl.load(open(pklFile))
    else:
        from adFVM import config
        from adFVM.mesh import Mesh
        from adFVM.field import Field, IOField
        case, time = sys.argv[1:3]
        time = float(time)
        mesh = Mesh.create(case)
        Field.setMesh(mesh)

        with IOField.handle(time):
            htc = IOField.read('htc_avg')
            Ma = IOField.read('Ma_avg')
            #htc = IOField.read('htc')
            #Ma = IOField.read('Ma')

            htc.partialComplete()
            Ma.partialComplete()
            #join = '/'
            #if config.hdf5:
            #    join = '_'
            #with open(mesh.getTimeDir(time) + join + 'wake_avg', 'r') as f:
            #    data = load(f)
            #    wakeCells, pl = data['arr_0'], data['arr_1']

        patches = ['pressure', 'suction']
       
        htc_args = []
        Ma_args = []

        nLayers = 200
        #nLayers = 1

        for patchID in patches:
            delta = -mesh.nInternalFaces + mesh.nInternalCells
            startFace = mesh.boundary[patchID]['startFace']
            nFaces = mesh.boundary[patchID]['nFaces']
            endFace = startFace + nFaces
            cellStartFace = startFace + delta
            cellEndFace = endFace + delta
            nFacesPerLayer = nFaces/nLayers

            #spanwise_average = lambda x: x.reshape((nFacesPerLayer, nLayers))[:,0]
            spanwise_average = lambda x: x.reshape((nFacesPerLayer, nLayers)).sum(axis=1)/nLayers
            x1 = spanwise_average(mesh.faceCentres[startFace:endFace, 0])
            x2 = spanwise_average(mesh.faceCentres[startFace:endFace, 1])
            x = (x1, x2)

            y = spanwise_average(htc.field[cellStartFace:cellEndFace])
            htc_args.extend([y, x])
            y = spanwise_average(Ma.field[cellStartFace:cellEndFace])
            Ma_args.extend([y, x])
        htc_args += [case + 'htc.pdf']
        Ma_args += [case + 'Ma.pdf']
        with open(pklFile, 'w') as f:
            pkl.dump([htc_args, Ma_args], f)

    match_velocity(*Ma_args)
    match_htc(*htc_args)

    #p0 = 175158.
    #nCellsPerLayer = len(wakeCells)/nLayers
    #y = pl[:nCellsPerLayer]/p0
    #x = 1000*mesh.cellCentres[wakeCells[:nCellsPerLayer], 1]
    #wake_args = [y, x, p0, case + 'wake.png']
    #match_wakes(*wake_args)

