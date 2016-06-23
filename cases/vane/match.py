#!/usr/bin/env python
from matplotlib import pyplot as plt, mlab
from matplotlib import markers as mk
import os
from profile import get_length, pressure, suction, c, pitch
from numpy import *
import csv
currdir = os.path.dirname(os.path.realpath(__file__))

def read_data(name):
    with open(currdir + '/' + name) as f:
        expe = array(list(csv.reader(f)))[:,:-1].astype(float)
    return expe

def match_htc(hp, coordsp, hs, coordss, saveFile):
    sp = get_length(pressure, coordsp)
    ss = get_length(suction, coordss)

    expe = read_data('data/htc_1.csv')

    fill = 1

    plt.scatter(expe[:,0], expe[:,1], c='k', alpha=fill,marker='o')
    plt.scatter(-sp*1000, hp, c='r', s=10, alpha=fill,marker='+')
    plt.scatter(ss*1000, hs, c='b', s=10, alpha=fill, marker='+')

    plt.savefig(saveFile)
    plt.clf()

def match_velocity(Map, coordsp, Mas, coordss, saveFile):

    sp = get_length(pressure, coordsp)
    ss = get_length(suction, coordss)

    expp = read_data('data/Ma_pressure_0.875.csv')
    exps = read_data('data/Ma_suction_0.875.csv')

    fill=1
    
    plt.scatter(expp[:,0], expp[:,1], c='r', marker='+')
    plt.scatter(exps[:,0], exps[:,1], c='b', marker='+')
    plt.scatter(sp/c, Map, c='r', s=10, alpha=fill, marker='o')
    plt.scatter(ss/c, Mas, c='b', s=10, alpha=fill, marker='o')
    plt.xlabel('s/c (mm)')
    plt.ylabel('Ma')

    plt.savefig(saveFile)
    plt.clf()

def match_wakes(pl, coords, saveFile):
    coords = (coords-coords.min())/pitch
    plt.scatter(coords, pl)
    plt.xlabel('y (mm)')
    plt.ylabel('pressure loss coeff')

    plt.savefig(saveFile)
    plt.clf()
