from sympy import *

x,y,z,t = symbols(('x', 'y', 'z', 't'))
gamma, mu, alpha = symbols(('gamma', 'mu', 'alpha'))
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

rho = sin(x) + sin(y) + sin(z)
rhoUx = sin(x) + sin(y) + sin(z)
rhoUy = sin(x) + sin(y) + sin(z)
rhoUz = 0
Ux = rhoUx/rho
Uy = rhoUy/rho
Uz = rhoUz/rho
U = [Ux, Uy, Uz]
rhoE = tan(z)
e = (rhoE-rho*(Ux*Ux + Uy*Uy + Uz*Uz)/2)
p = (gamma-1)*e

# euler
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

# navier stokes
SrhoUx += -mu*div(sigma[0])
SrhoUy += -mu*div(sigma[1])
SrhoUz += -mu*div(sigma[2])
SrhoE += -alpha*lap(e) -mu*div(sigmadotU)

print Srho
print SrhoE
print SrhoUx
def Source():
    pass
