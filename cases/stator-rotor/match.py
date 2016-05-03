#!/usr/bin/env python
from matplotlib import pyplot as plt, mlab
from matplotlib import markers as mk
import os
from profile import get_length, pressure, suction, c, pitch
from numpy import *

def match_htc(hp, coordsp, hs, coordss, saveFile):
    sp = get_length(pressure, coordsp)
    ss = get_length(suction, coordss) 

    fill = 1

    plt.scatter(-sp*1000, hp, c='r', s=10, alpha=fill,marker='+')
    plt.scatter(ss*1000, hs, c='b', s=10, alpha=fill, marker='+')

    plt.savefig(saveFile)
    plt.clf()

def match_velocity(Map, coordsp, Mas, coordss, saveFile):
    sp = get_length(pressure, coordsp)
    ss = get_length(suction, coordss)

    fill=1

    plt.scatter(sp/c, Map, c='r', s=10, alpha=fill)
    plt.scatter(ss/c, Mas, c='b', s=10, alpha=fill)

    plt.savefig(saveFile)
    plt.clf()

def match_wakes(pl, coords, saveFile):
    coords = (coords-coords.min())/pitch
    plt.scatter(coords, pl)
    plt.savefig(saveFile)
