#!/usr/bin/python2
from __future__ import print_function
import numpy as np
import numpad as ad
import time

from mesh import Mesh
from field import Field, FaceField
from ops import interpolate, div, ddt, solve, laplacian, grad

gamma = 1.4
mu = 2.5e-5
Pr = 0.7
alpha = mu/Pr

case = 'test/'
mesh = Mesh(case)

t = 0.1
dt = 0.005

#initialize
rho = Field.zeros('rho', mesh, 1)
rhoU = Field.zeros('rhoU', mesh, 3)
rhoE = Field.zeros('rhoE', mesh, 1)


for i in range(0, 300):
    if i % 20 == 0:
        rho.write(t)

    print('Simulation Time:', t, 'Time step:', dt)

    rho0 = Field.copy(rho)
    rhoU0 = Field.copy(rhoU)
    rhoE0 = Field.copy(rhoE)

    def eq(rho, rhoU, rhoE):

        # implement upwind/central
        rhof = interpolate(rho)
        rhoUf = interpolate(rhoU)
        rhoEf = interpolate(rhoE)

        U = rhoU/rho
        Uf = rhoUf/rhof
        Ef = rhoEf/rhof
        ef = Ef - 0.5*Uf.mag()
        pf = (gamma - 1)*ef

        return [ddt(rho, rho0, dt) + div(rhof, Uf),
                ddt(rhoU, rhoU0, dt) + div(rhoUf, Uf) + grad(pf) - (laplacian(U, mu)), #+ div(grad(U))
                ddt(rhoE, rhoE0, dt) + div(rhoEf + pf, Uf) - (laplacian(ef, alpha) )] #+ div(sigma, Uf))]
    
    solve(eq, [rho, rhoU, rhoE])

    t += dt
    t = round(t, 6)

    print()
