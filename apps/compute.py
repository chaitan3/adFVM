#!/usr/bin/python2
import time as timer
import argparse

from adFVM import config
from adFVM.parallel import pprint
from adFVM.postpro import *
from adFVM.density import RCF
from adFVM.field import IOField

parser = argparse.ArgumentParser()
parser.add_argument('case')
parser.add_argument('-t', '--times', nargs='+', type=float)
user = parser.parse_args(config.args)
times = user.times

names = ['gradrho', 'gradU', 'gradp', 'gradc', 'divU', 'enstrophy', 'Q']
dimensions = [(3,), (3,3), (3,),(3,),(1,), (1,), (1,)]

solver = RCF(user.case)
mesh = solver.mesh
if not times:
    times = mesh.getTimes()

solver.readFields(times[0])
solver.compileInit()
computer = computeGradients(solver)

if config.compile:
    exit()

for index, time in enumerate(times):
    pprint('Time:', time)
    start = timer.time()
    rho, rhoU, rhoE = solver.initFields(time)
    U, T, p = solver.U, solver.T, solver.p

    IOField.openHandle(time)

    outputs = computer(U.field, T.field, p.field)
    outputsF = []
    for field, name, dim in zip(outputs, names, dimensions):
        outputsF.append(IOField(name, field, dim))
        if len(dim) != 2:
            outputsF[-1].write()
    pprint()
    # adjoint blowup
    fields = getAdjointNorm(rho, rhoU, rhoE, U, T, p, *outputs)
    for phi in fields:
        phi.write()#, skipProcessor=True)
    pprint()
    enstrophy, Q = getEnstrophyAndQ(outputsF[1])
    enstrophy.write(name='enstrophy') 
    Q.write(name='Q')

    c, M, pt, s = getTotalPressureAndEntropy(U, T, p, solver)
    c.write(name='c') 
    M.write(name='Ma')
    pt.write(name='pt')
    s.write(name='s')
    pprint()

    ## rhoaByV
    #rhoa = IOField.read('rhoa')
    #rhoaByV = getFieldByVolume(rhoa)
    #rhoaByV.write()
    #pprint()

    ## adjoint energy
    #rhoUa = IOField.read('rhoUa')
    #rhoEa = IOField.read('rhoEa')
    #adjEnergy = getAdjointEnergy(solver, rhoa, rhoUa, rhoEa)
    #pprint('L2 norm adjoint', time, adjEnergy)
    #pprint()

    IOField.closeHandle()

    end = timer.time()
    pprint('Time for computing: {0}'.format(end-start))

    pprint()
