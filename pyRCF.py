#!/usr/bin/python2
from __future__ import print_function
import numpy as np
import numpad as ad
import time

from mesh import Mesh
from field import Field, FaceField
from ops import  div, ddt, laplacian, grad, implicit, explicit
from ops import interpolate, upwind

gamma = 1.4
#mu = 2.5e-5
mu = 0.
Pr = 0.7
alpha = mu/Pr
Cp = 1007
Cv = Cp/gamma


case = 'shockTube/'
mesh = Mesh(case)

t = 0
dt = 1e-4
writeInterval = 5

#initialize
pos = FaceField('pos', mesh, ad.ones((mesh.nFaces, 1)))
neg = FaceField('neg', mesh, ad.ones((mesh.nFaces, 1)))

p = Field.read('p', mesh, t)
T = Field.read('T', mesh, t)
U = Field.read('U', mesh, t)
e = Cv*T

def primitive(rho, rhoU, rhoE):
    U = rhoU/rho
    E = rhoE/rho
    e = E - 0.5*U.mag()
    p = (gamma-1)*rho*e
    return U, e, p

def conservative(U, e, p):
    rho = p/(e*(gamma-1))
    E = e + 0.5*U.mag()
    return rho, rho*U, rho*E


rho, rhoU, rhoE = conservative(U, e, p)
rho.name, rhoU.name, rhoE.name = 'rho', 'rhoU', 'rhoE'

for timeIndex in range(1, 300):

    print('Simulation Time:', t, 'Time step:', dt)

    rho0 = Field.copy(rho)
    rhoU0 = Field.copy(rhoU)
    rhoE0 = Field.copy(rhoE)

    def eq(rho, rhoU, rhoE):

        rhoLF, rhoRF = upwind(rho, pos), upwind(rho, neg) 
        rhoULF, rhoURF = upwind(rhoU, pos), upwind(rhoU, neg) 
        rhoELF, rhoERF = upwind(rhoE, pos), upwind(rhoE, neg) 

        ULF, eLF, pLF = primitive(rhoLF, rhoULF, rhoELF)
        URF, eRF, pRF = primitive(rhoRF, rhoURF, rhoERF)
        U, e, p = primitive(rho, rhoU, rhoE)

        cLF, cRF = (gamma*pLF/rhoLF)**0.5, (gamma*pRF/rhoRF)**0.5
        UnLF, UnRF = ULF.dotN(), URF.dotN()
        cF = (cLF + UnLF, cRF + UnRF, cLF - UnLF, cRF - UnRF)
        #wtf? adjoint? if not, make it readable
        aF = -FaceField('aF', mesh, ad.adarray(np.max(ad.value(ad.hstack([x.field for x in cF])), axis=1)).reshape((-1,1)))

        rhoFlux = 0.5*(rhoLF*UnLF + rhoRF*UnRF) - 0.5*aF*(rhoLF-rhoRF)
        rhoUFlux = 0.5*(rhoULF*UnLF + rhoURF*UnRF) - 0.5*aF*(rhoULF-rhoURF)
        rhoEFlux = 0.5*((rhoELF + pLF)*UnLF + (rhoERF + pRF)*UnRF) - 0.5*aF*(rhoELF-rhoERF)
        pF = 0.5*(pLF + pRF)

        return [ddt(rho, rho0, dt) + div(rhoFlux),
                ddt(rhoU, rhoU0, dt) + div(rhoUFlux) + grad(pF) - (laplacian(U, mu)), #+ div(grad(U))
                ddt(rhoE, rhoE0, dt) + div(rhoEFlux) - (laplacian(e, alpha) )] #+ div(sigma, Uf))]
    
    #implicit(eq, [rho, rhoU, rhoE])
    explicit(eq, [rho, rhoU, rhoE])

    t += dt
    t = round(t, 6)
    print()

    if timeIndex % writeInterval == 0:
        rho.write(t)
        rhoU.write(t)
        rhoE.write(t)
   
