from sympy import *
from sympy.utilities.lambdify import lambdify
import numpy as np

from field import Field, CellField

x,y,z,t = symbols(('x', 'y', 'z', 't'))
gamma, Cv, Pr = symbols(('gamma', 'Cv', 'Pr'))
X = [x, y, z]

def div(phi, U=None):
    if U is None:
        return sum([diff(p, d) for p,d in zip(phi, X)])
    else:
        return sum([diff(phi*u, d) for u,d in zip(U, X)])
def dot(U, V):
    return sum([u*v for u,v in zip(U, V)])
def lap(phi):
    return sum([diff(diff(phi, d), d) for d in X])

# manufactured solution
rho = 1 + 0.1*sin(0.75*pi*x) + 0.15*cos(1.0*pi*y) + 0*sin(pi*z)
rhoUx = 70 + 4*sin(1.66*pi*x) + -12*cos(1.5*pi*y) + 0*cos(pi*z)
rhoUy = 90 + -20*cos(1.5*pi*x) + 4*sin(1.0*pi*y) + 0*sin(pi*z)
rhoUz = 0*rho
rhoE = 3e5 + -0.3e5*cos(1.0*pi*x) + 0.2e5*sin(1.25*pi*y) + 0*cos(pi*z)

Ux = rhoUx/rho
Uy = rhoUy/rho
Uz = rhoUz/rho
U = [Ux, Uy, Uz]
e = (rhoE-rho*(Ux*Ux + Uy*Uy + Uz*Uz)/2)
T = e/Cv
p = (gamma-1)*e

# euler, simplify?
Srho = diff(rho, t) + div(rho, U)
SrhoUx = diff(rhoUx, t) + div(rhoUx, U) + diff(p, x)
SrhoUy = diff(rhoUy, t) + div(rhoUy, U) + diff(p, y)
SrhoUz = diff(rhoUz, t) + div(rhoUz, U) + diff(p, z)
SrhoE = diff(rhoE, t) + div(rhoE + p, U)

sigma = []
for i in range(0, 3):
    sigma.append([])
    for j in range(0, 3):
        sigma[-1].append(diff(U[i], X[j]) + diff(U[j], X[i]))
trace = sum([sigma[i][i]/2 for i in range(0, 3)])
for i in range(0, 3):
    sigma[i][i] -= 2./3*trace
sigmadotU = [dot(sig, U) for sig in sigma]
mu = 10.
kappa = mu*Cv*gamma/Pr

# navier stokes
SrhoUx += -mu*div(sigma[0])
SrhoUy += -mu*div(sigma[1])
SrhoUz += -mu*div(sigma[2])
SrhoE += -kappa*lap(T) -mu*div(sigmadotU)

print Srho
print SrhoUx
print SrhoUy
print SrhoUz
print SrhoE

def source(solver):
    cellCentres = solver.mesh.cellCentres[:solver.mesh.nInternalCells]
    X = cellCentres[:, 0]
    Y = cellCentres[:, 1]
    Z = cellCentres[:, 2]
    subs={Cv:solver.Cv, Pr:solver.Pr, gamma:solver.gamma, t: solver.t}
    func = lambdify((x, y, z), [Srho.subs(subs), SrhoUx.subs(subs), SrhoUy.subs(subs), SrhoUz.subs(subs), SrhoE.subs(subs)], np)
    res = func(X, Y, Z)
    # rhoUz 0 hack
    res[3] = X*0
    Frho = Field('Srho', res[0].reshape(-1,1))
    FrhoU = Field('SrhoU', np.column_stack(res[1:4]))
    FrhoE = Field('SrhoE', res[4].reshape(-1,1))
    return [Frho, FrhoU, FrhoE]

def solution(T, mesh):
    cellCentres = mesh.cellCentres[:mesh.nInternalCells]
    X = cellCentres[:, 0]
    Y = cellCentres[:, 1]
    Z = cellCentres[:, 2]
    subs = {t: T}
    func = lambdify((x, y, z), [rho.subs(subs), rhoUx.subs(subs), rhoUy.subs(subs), rhoUz.subs(subs), rhoE.subs(subs)], np)
    res = func(X, Y, Z)
    # rhoUz 0 hack
    res[3] = X*0
    Frho = CellField('rho', res[0].reshape(-1,1))
    FrhoU = CellField('rhoU', np.column_stack(res[1:4]))
    FrhoE = CellField('rhoE', res[4].reshape(-1,1))
    Frho.write(1.0)
    FrhoU.write(1.0)
    FrhoE.write(1.0)
    return [Frho, FrhoU, FrhoE]


