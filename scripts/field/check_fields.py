#!/usr/bin/python2
import sys
import numpy as np

from adFVM.density import RCF
from adFVM.field import IOField

case, time = sys.argv[1:3]
time = float(time)
rcf = RCF(case)
with IOField.handle(time):
    U = IOField.read('U')
    T = IOField.read('T')
    p = IOField.read('p')
rho, rhoU, rhoE = rcf.conservative(U, T, p)
fields = [U, T, p, rho, rhoU, rhoE]

for phi in fields:
    phi.info()
