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
pos = FaceField('pos', mesh, ad.adarray(mesh.normals))
neg = FaceField('neg', mesh, ad.adarray(-mesh.normals))

p = Field.read('p', mesh, t)
T = Field.read('T', mesh, t)
U = Field.read('U', mesh, t)
print()

def primitive(rho, rhoU, rhoE):
    U = rhoU/rho
    E = rhoE/rho
    e = E - 0.5*U.magSqr()
    p = (gamma-1)*rho*e
    T = e*(1./Cv)
    U.name, T.name, p.name = 'U', 'T', 'p'
    return U, T, p

def conservative(U, T, p):
    e = Cv*T
    rho = p/(e*(gamma-1))
    E = e + 0.5*U.magSqr()
    rhoU = rho*U
    rhoE = rho*E
    rho.name, rhoU.name, rhoE.name = 'rho', 'rhoU', 'rhoE'
    return rho, rhoU, rhoE


rho, rhoU, rhoE = conservative(U, T, p)

for timeIndex in range(1, 300):

    print('Simulation Time:', t, 'Time step:', dt)

    def eq(rhoC, rhoUC, rhoEC):

        rhoLF, rhoRF = upwind(rhoC, pos), upwind(rhoC, neg) 
        rhoULF, rhoURF = upwind(rhoUC, pos), upwind(rhoUC, neg) 
        rhoELF, rhoERF = upwind(rhoEC, pos), upwind(rhoEC, neg) 

        ULF, TLF, pLF = primitive(rhoLF, rhoULF, rhoELF)
        URF, TRF, pRF = primitive(rhoRF, rhoURF, rhoERF)
        U, T, p = primitive(rhoC, rhoUC, rhoEC)
        e = Cv*T

        cLF, cRF = (gamma*pLF/rhoLF)**0.5, (gamma*pRF/rhoRF)**0.5
        UnLF, UnRF = ULF.dotN(), URF.dotN()
        cF = (cLF + UnLF, cRF + UnRF, cLF - UnLF, cRF - UnRF)
        #wtf? adjoint? if not, make it readable
        aF = FaceField('aF', mesh, ad.adarray(np.max(ad.value(ad.hstack([x.field for x in cF])), axis=1)).reshape((-1,1)))


        rhoFlux = 0.5*(rhoLF*UnLF + rhoRF*UnRF) - 0.5*aF*(rhoRF-rhoLF)
        rhoUFlux = 0.5*(rhoULF*UnLF + rhoURF*UnRF) - 0.5*aF*(rhoURF-rhoULF)
        rhoEFlux = 0.5*((rhoELF + pLF)*UnLF + (rhoERF + pRF)*UnRF) - 0.5*aF*(rhoERF-rhoELF)
        pF = 0.5*(pLF + pRF)
        #FaceField('gradpF', mesh, grad(pF)).info()
        #time.sleep(1)

        return [ddt(rhoC, rho, dt) + div(rhoFlux),
                ddt(rhoUC, rhoU, dt) + div(rhoUFlux) + grad(pF) - (laplacian(U, mu)), #+ div(grad(U))
                ddt(rhoEC, rhoE, dt) + div(rhoEFlux) - (laplacian(e, alpha) )] #+ div(sigma, Uf))]
    
    #implicit(eq, [rho, rhoU, rhoE])
    explicit(eq, [rho, rhoU, rhoE], dt)

    t += dt
    t = round(t, 6)
    if timeIndex % writeInterval == 0:
        U, T, p = primitive(rho, rhoU, rhoE)
        rho.write(t)
        rhoU.write(t)
        rhoE.write(t)
        U.write(t)
        T.write(t)
        p.write(t)

    print()
   
