#!/usr/bin/python2
import argparse

from adFVM import config, parallel
from adFVM.config import T
from adFVM.density import RCF

parser = argparse.ArgumentParser()
parser.add_argument('case')
parser.add_argument('time', type=float)
parser.add_argument('-i', '--timeIntegrator', required=False, default=RCF.defaultConfig['timeIntegrator'])
parser.add_argument('-l', '--CFL', required=False, default=RCF.defaultConfig['CFL'], type=float)
parser.add_argument('--Cp', required=False, default=RCF.defaultConfig['Cp'], type=float)
parser.add_argument('--riemann', required=False, default=RCF.defaultConfig['riemannSolver'])
parser.add_argument('--lts', action='store_true')
parser.add_argument('-v', '--inviscid', action='store_true')
parser.add_argument('--dynamic', action='store_true')
mu = RCF.defaultConfig['mu']

parser.add_argument('-n', '--nSteps', required=False, default=10000, type=int)
parser.add_argument('-w', '--writeInterval', required=False, default=500, type=int)
parser.add_argument('--dt', required=False, default=1e-9, type=float)
user = parser.parse_args(config.args)
if user.inviscid:
    mu = lambda T: config.VSMALL*T
solver = RCF(user.case, mu=mu, timeIntegrator=user.timeIntegrator, CFL=user.CFL, Cp=user.Cp, riemannSolver=user.riemann, dynamicMesh=user.dynamic, localTimeStep=user.lts)
solver.readFields(user.time)
solver.compile()
solver.run(startTime=user.time, dt=user.dt, nSteps=user.nSteps, writeInterval=user.writeInterval)

# for profiling purposes
if parallel.rank == 0 and config.user.profile:
    T.compile.profiling._atexit_print_fn()
