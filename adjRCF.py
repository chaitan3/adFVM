import numpy as np

from config import ad
from field import CellField
from op import div, laplacian, grad
from interp import central

#BCs
class AdjRCF(object):
    def __init__(self, primal):
        self.primal = primal
        self.mesh = primal.mesh
        self.stackedSymAdjointFields = None

        stackedFields = ad.matrix()
        stackedAdjointFields = ad.matrix()
        stackedSymAdjointFields = ad.matrix()
        dt = ad.scalar()
        self.internal_gradient = primal.function([stackedFields, stackedAdjointFields, stackedSymAdjointFields, dt], self.compute(stackedAdjointFields, stackedSymAdjointFields, stackedSymAdjointFields, dt), 'cont_adjoint')
        self.stackedSymAdjointFields = None

    def gradient(self, stackedFields, stackedAdjointFields, dt):
        if self.stackedSymAdjointFields is None:
            self.stackedSymAdjointFields = stackedAdjointFields.copy()
        newStackedAdjointFields, self.stackedSymAdjointFields = self.internal_gradient(stackFields, stackedAdjointFields, stackedSymAdjointFields, dt)
        return newStackedAdjointFields
        
    def compute(self, stackedFields, stackedAdjointFields, stackedSymAdjointFields, dt):
        primal = self.primal
        mesh = self.mesh
        #padding
        paddedStackedFields = primal.padField(stackedFields)
        paddedStackedSymAdjointFields = primal.padField(stackedSymAdjointFields)
        symaP, symUaP, symEaP = primal.unstackFields(paddedStackedSymAdjointFields, CellField)
        rhoa, rhoUa, rhoEa = primal.unstackFields(stackedAdjointFields, CellField)
        syma = CellField.getOrigField(symaP)
        symUa = CellField.getOrigField(symUaP)
        symEa = CellField.getOrigField(symEaP)
        rhoP, rhoUP, rhoEP = primal.unstackFields(paddedStackedFields, CellField) 
        rho = CellField.getOrigField(rhoP)
        rhoU = CellField.getOrigField(rhoUP)
        rhoE = CellField.getOrigField(rhoEP)
        
        U, T, p = primal.primitive(rho, rhoU, rhoE)
        g = primal.gamma
        g1 = g-1
        sg = np.sqrt(g)
        sg1 = np.sqrt(g1)
        #define a, b, c
        c = (g*p/rho).sqrt()
        bI = (c*(1./sg)).internalField()
        aI = (c*(sg/sg1)).internalField()
        rhoI = rho.internalField()
        pI = p.internalField()
        #define aF, bF, cF
        #better interpolation?
        TF = central(T, mesh)
        cF = central(c, mesh)
        UF = central(U, mesh)
        bF = cF*(1./sg)
        aF = cF*(sg/sg1)
        UnF = UF.dotN()

        #define gradaI, gradpI, gradrhoI
        divUI = div(UnF)
        gradUI = grad(UF)
        gradrhoI = grad(central(rho, mesh))
        gradpI = grad(central(p, mesh))
        gradaI = grad(aF)

        #define muF, alphaF
        muF = primal.mu(TF)
        alphaF = primal.kappa(muF, TF)*(1./primal.Cp)

        #define F for syma's, use central
        symaF = central(syma, mesh)
        symUaF = central(symUa, mesh)
        symUanF = symUaF.dotN()
        symEaF = central(symEa, mesh)
        symaFlux = UnF*symaF + bF*symUanF
        symUaFlux = bF*symaF*mesh.Normals + UnF*symUaF + aF*symEaF*mesh.Normals
        symEaFlux = aF*symUanF + UnF*symEaF

        symaI = syma.internalField()
        symUaI = symUa.internalField()
        symEaI = symEa.internalField()
        symaSource = (bI/rhoI)*gradrhoI.dot(symUaI) + 0.5*sg1*divUI*symEaI
        #check correctness of dot product
        symUaSource = gradUI.dot(symUaI) + 0.5*(aI/pI)*gradpI*symEaI
        symEaSource = (2/g1)*gradaI.dot(symUaI) + 0.5*g1*divUI*symEaI

        #viscous, check div?
        divSymUaF = central(div(symUanF), mesh)
        symUaVisc = laplacian(symUa, muF) + grad((muF*0.3333)*divSymUaF)
        symEaVisc = laplacian(symEa, alphaF)
        
        #time step
        dsyma = dt*(div(symaFlux) + symaSource)
        dsymUa = dt*(div(symUaFlux) + symUaSource + symUaVisc)
        dsymEa = dt*(div(symEaFlux) + symEaSource + symEaVisc)

        dprima = c/(sg*rho)*(dsyma - (1./sg1)*dsymEa)
        dprimUa = dsymUa
        dprimEa = (sg/sg1)*dsymEa/(rho*c)

        #update
        syma.setInternalField(dsyma.field)
        symUa.setInternalField(dsymUa.field)
        symEa.setInternalField(dsymEa.field)
        rhoaI = rhoa.internalField() + dprima + (U/rho).dot(dprimUa) + 0.5*g1*U.dot(U)*dprimEa
        rhoUaI = rhoUa.internalField() + dprimUa/rho - g1*U*dprimEa
        rhoEaI = rhoEa.internalField() + g1*dprimEa
        #boundary condition transformation?
        #BC fields work
        rhoa.setInternalField(rhoaI.field)
        rhoUa.setInternalField(rhoUaI.field)
        rhoEa.setInternalField(rhoEaI.field)

        return primal.stackFields([rhoa, rhoUa, rhoEa], ad), \
        primal.stackFields([syma, symUa, symEa], ad)

