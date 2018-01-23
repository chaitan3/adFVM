#!/usr/bin/python2
import time as timer
import argparse

import adFVM
from adFVM.parallel import pprint
from adFVM.postpro import *
from adFVM.density import RCF
from adFVM.field import IOField

parser = argparse.ArgumentParser()
parser.add_argument('case')
parser.add_argument('-t', '--times', nargs='+', type=float)
user = parser.parse_args(adFVM.config.args)
times = user.times

names = ['gradT', 'gradrho', 'gradU', 'gradp', 'gradc', 'divU', 'enstrophy', 'Q']
dimensions = [(3,),(3,), (3,3), (3,),(3,),(1,), (1,), (1,)]

solver = RCF(user.case)
mesh = solver.mesh
if not times:
    times = mesh.getTimes()

solver.readFields(times[0])
solver.compile()

for index, time in enumerate(times):
    pprint('Time:', time)
    start = timer.time()

    fields = solver.readFields(time)  
    rho, rhoU, rhoE = fields
    #rho, rhoU, rhoE = solver.initFields(fields)
    rho.defaultComplete()
    rhoU.defaultComplete()
    rhoE.defaultComplete()
    U, T, p = solver.primitive(rho, rhoU, rhoE)

    with IOField.handle(time):

        outputs = computeGradients(solver, U, T, p)
        outputsF = []
        for field, name, dim in zip(outputs, names, dimensions):
            outputsF.append(IOField(name, field, dim))
            if len(dim) != 2:
                outputsF[-1].defaultComplete()
                outputsF[-1].write()
        pprint()

        #Re = getRe(U, T, p, rho, 2.5e-4)
        ##Re = getRe(U, T, p, rho, 5.5e-3)
        #Re.write(name='Re')
        #exit(1)
        ##pprint(Re.getPatch('outlet').mean())

        #enstrophy, Q = getEnstrophyAndQ(outputsF[2])
        #enstrophy.write(name='enstrophy') 
        #Q.write(name='Q')
        #exit(1)

        #c, M, pt, s = getTotalPressureAndEntropy(U, T, p, solver)
        ##c.write(name='c') 
        ##M.write(name='Ma')
        #pt.write(name='pt')
        ##s.write(name='s')
        #pprint()

        #uplus, yplus, _, _ = getYPlus(U, T, rho, ['airfoil'])
        #uplus = IOField.boundaryField('uplus', uplus, (3,))
        #yplus = IOField.boundaryField('yplus', yplus, (1,))
        ##print yplus.field.max(), yplus.field.min()
        #uplus.write()
        #yplus.write()

        ## adjoint norm
        #rhoa = IOField.read('rhoa')
        #rhoUa = IOField.read('rhoUa')
        #rhoEa = IOField.read('rhoEa')
        #rhoa = IOField.read('rhoa')
        #rhoUa = IOField.read('rhoUa')
        #rhoEa = IOField.read('rhoEa')
        rhoa = rhoUa = rhoEa = None

        ###scale = lambda x: 1/(1+np.exp(-10*(x/parallel.max(x)-1)))
        scale = None
        for visc in ["abarbanel", "entropy_jameson", "uniform", "entropy_hughes"]:
        #for visc in ["abarbanel", "entropy_jameson", "entropy_hughes"]:
            adjNorm, energy, diss = getAdjointMatrixNorm(rhoa, rhoUa, rhoEa, rho, rhoU, rhoE, U, T, p, *outputs, visc=visc, scale=scale)
            adjNorm.write()
            #energy.write()
            #diss.write()
            pprint()

        #adjEnergy = getAdjointEnergy(solver, rhoa, rhoUa, rhoEa)
        #pprint('L2 norm adjoint', time, adjEnergy)
        #pprint()

        ## adjoint viscosity
        #mua = getAdjointViscosity(rho, rhoU, rhoE, 1e-2, outputs=outputs, init=False)
        #mua.write()

    end = timer.time()
    pprint('Time for computing: {0}'.format(end-start))

    pprint()
