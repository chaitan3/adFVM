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
