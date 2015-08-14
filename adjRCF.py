import numpy as np

from field import CellField
from op import div

#BCs

def gradient(stackedFields, stackedAdjointFields):
    mesh = primal.mesh
    rhoa, rhoUa, rhoEa = primal.unstackFields(stackedAdjointFields, CellField)
    rho, rhoU, rhoE = primal.unstackFields(stackedFieldS) 

    g = primal.gamma
    g1 = g-1
    sg = np.sqrt(g)
    sg1 = np.sqrt(g1)
    #define UnF
    #define a, b, c
    #define aF, bF, cF
    cF = 
    bF = cF/sg
    aF = cF*sg/sg1
    divU = div(UnF)
    #define grada, gradp, gradrho
    
    symaFlux = UnF*symaF + bF*symUnF
    symUaFlux = bF*symaF*mesh.Normals + UnF*symUF + aF*symEaF*mesh.Normals
    symEaFlux = aF*symUnF + UnF*symEaF

    symaSource = (b/rho)*gradrho.dot(symUa) + 0.5*sg1*divU*symEa
    #check correctness of dot product
    symUaSource = gradU.dot(symUa) + 0.5*(a/p)*gradp*symEa
    symEaSource = (2/g1)*grada.dot(symUa) + 0.5*g1*divU*symEa

    #viscous
    
    #time step
    dsyma = dt*(div(symaFlux) + symaSource)
    dsymUa = dt*(div(symUaFlux) + symUaSource)
    dsymEa = dt*(div(symEaFlux) + symEaSource)
    dprima = c/(sg*rho)*(dsyma - (1./sg1)*dsymEa)
    dprimUa = dsymUa
    dprimEa = (sg/sg1)*dsymEa/(rho*c)

    #update
    syma += dsyma
    symUa += dsymUa
    symEa += dsymEa
    rhoa += dprima + (U/rho).dot(dprimUa) + 0.5*g1*U.dot(U)*dprimEa
    rhoUa += dprimUa/rho - g1*U*dprimEa
    rhoEa += g1*dprimEa

    newStackedAdjointFields = primal.stackFields([rhoa, rhoUa, rhoEa], ad) 
    return newStackedAdjointFields

