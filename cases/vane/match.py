#!/usr/bin/env python
from matplotlib import pyplot as plt, mlab
from matplotlib import markers as mk
import os
from profile import get_length, pressure, suction, c
from numpy import *

def match_htc(hp, coordsp, hs, coordss, saveFile):
    im = plt.imread('/home/talnikar/Dropbox/Research/results/2014/turbine_blade_verification/blade-htc1.png')
    plt.imshow(im)

    def transx(a):
        return 43 + (373-43)*((a+100)/200)
    def transy(a):
        return 283 - (283-15)*(a/1600)

    sp = get_length(pressure, coordsp)
    ss = get_length(suction, coordss)

    fill = 1

    plt.scatter(transx(-sp*1000), transy(hp), c='r', s=10, alpha=fill,marker='+')
    plt.scatter(transx(ss*1000), transy(hs), c='b', s=10, alpha=fill, marker='+')

    plt.savefig(saveFile)
    plt.clf()

def match_velocity(Map, coordsp, Mas, coordss, saveFile):
    im = plt.imread('/home/talnikar/Dropbox/Research/results/2014/turbine_blade_verification/blade-velocity.png')
    plt.imshow(im)

    def transx(a):
        return 77 + (684-77)*(a/1.4)
    def transy(a):
        return 461 - (461-32)*(a/2.0)

    sp = get_length(pressure, coordsp)
    ss = get_length(suction, coordss)

    fill=1

    plt.scatter(transx(sp/c), transy(Map), c='r', s=10, alpha=fill)
    plt.scatter(transx(ss/c), transy(Mas), c='b', s=10, alpha=fill)

    plt.savefig(saveFile)
    plt.clf()
