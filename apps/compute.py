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
solver.compile()

for index, time in enumerate(times):
    pprint('Time:', time)
    start = timer.time()

    fields = solver.readFields(time)  
    rho, rhoU, rhoE = solver.initFields(fields)
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


        #enstrophy, Q = getEnstrophyAndQ(outputsF[1])
        #enstrophy.write(name='enstrophy') 
        #Q.write(name='Q')

        #c, M, pt, s = getTotalPressureAndEntropy(U, T, p, solver)
        #c.write(name='c') 
        #M.write(name='Ma')
        #pt.write(name='pt')
        #s.write(name='s')
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
        rhoa = IOField.read('rhoa')
        rhoUa = IOField.read('rhoUa')
        rhoEa = IOField.read('rhoEa')

        #scale = lambda x: 1/(1+np.exp(-10*(x/parallel.max(x)-1)))
        scale = None
        for visc in ["abarbanel", "entropy", "uniform"]:
            adjNorm, energy, diss = getAdjointMatrixNorm(rhoa, rhoUa, rhoEa, rho, rhoU, rhoE, U, T, p, *outputs, visc=visc, scale=scale)
            adjNorm.write()
            energy.write()
            diss.write()
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
