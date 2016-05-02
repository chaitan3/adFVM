from pyRCF import RCF
from compute import getHTC, getIsentropicMa

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('case')
parser.add_argument('time', nargs='+', type=float)
user = parser.parse_args(config.args)

solver = RCF(user.case)
mesh = solver.mesh
solver.initialize(user.time[0])

for time in user.time:
    rho, rhoU, rhoE = solver.initFields(time)
    U, T, p = solver.U, solver.T, solver.p

    htc = getHTC(T, 420., ['pressure', 'suction'])
    Ma = getIsentropicMa(p, 171325.)
    
    
