#!/usr/bin/python2
import argparse
import numpy as np

from adFVM import config, parallel
from adFVM.density import RCF

parser = argparse.ArgumentParser()
parser.add_argument('case')
parser.add_argument('time', type=float)
parser.add_argument('-i', '--timeIntegrator', required=False, default=RCF.defaultConfig['timeIntegrator'])
parser.add_argument('-l', '--CFL', required=False, default=RCF.defaultConfig['CFL'], type=float)
parser.add_argument('--Cp', required=False, default=RCF.defaultConfig['Cp'], type=float)
parser.add_argument('--riemann', required=False, default=RCF.defaultConfig['riemannSolver'])
parser.add_argument('-f', '--reconstructor', required=False, default=RCF.defaultConfig['faceReconstructor'])
parser.add_argument('--lts', action='store_true')
parser.add_argument('-v', '--inviscid', action='store_true')
parser.add_argument('--dynamic', action='store_true')
mu = RCF.defaultConfig['mu']

parser.add_argument('-n', '--nSteps', required=False, default=10000, type=int)
parser.add_argument('-w', '--writeInterval', required=False, default=500, type=int)
parser.add_argument('--dt', required=False, default=1e-9, type=float)
parser.add_argument('-t', '--endTime', required=False, default=np.inf, type=float)
user = parser.parse_args(config.args)
if user.inviscid:
    mu = lambda T: 0.*T
solver = RCF(user.case, mu=mu, timeIntegrator=user.timeIntegrator, CFL=user.CFL, Cp=user.Cp, riemannSolver=user.riemann, dynamicMesh=user.dynamic, localTimeStep=user.lts, faceReconstructor=user.reconstructor)
solver.readFields(user.time)
solver.compile()
solver.run(startTime=user.time, endTime=user.endTime, dt=user.dt, nSteps=user.nSteps, writeInterval=user.writeInterval)
