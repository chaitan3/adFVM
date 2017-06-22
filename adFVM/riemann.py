from . import config
from .tensor import Tensor
from .field import Field

def eulerLaxFriedrichs(gamma, pLF, pRF, TLF, TRF, ULF, URF, \
                rhoLF, rhoRF, rhoULF, rhoURF, rhoELF, rhoERF, Normals):

    UnLF, UnRF = ULF.dot(Normals), URF.dot(Normals)
    cLF = (gamma*pLF/rhoLF).sqrt()
    cRF = (gamma*pRF/rhoRF).sqrt()
    aF = Field.max(UnLF.abs() + cLF, UnRF.abs() + cRF)
    #Z = Field('Z', ad.bcalloc(config.precision(0.), (mesh.nFaces, 1)), (1,))
    #apF = Field.max(Field.max(UnLF + cLF, UnRF + cRF), Z)
    #amF = Field.min(Field.min(UnLF - cLF, UnRF - cRF), Z)
    #aF = Field.max(apF.abs(), amF.abs())

    rhoFlux = 0.5*(rhoLF*UnLF + rhoRF*UnRF) - 0.5*aF*(rhoRF-rhoLF)
    rhoUFlux = 0.5*(rhoULF*UnLF + rhoURF*UnRF + (pLF + pRF)*Normals) - 0.5*aF*(rhoURF-rhoULF)
    rhoEFlux = 0.5*((rhoELF + pLF)*UnLF + (rhoERF + pRF)*UnRF) - 0.5*aF*(rhoERF-rhoELF)

    return rhoFlux, rhoUFlux, rhoEFlux


def eulerRoe(gamma, pLF, pRF, TLF, TRF, ULF, URF, \
                rhoLF, rhoRF, rhoULF, rhoURF, rhoELF, rhoERF, Normals):

    rhoUnLF, rhoUnRF = rhoLF*ULF.dot(Normals), rhoRF*URF.dot(Normals)
    hLF = gamma*pLF/((gamma-1)*rhoLF) + 0.5*ULF.magSqr()
    hRF = gamma*pRF/((gamma-1)*rhoRF) + 0.5*URF.magSqr()

    rhoFlux = 0.5*(rhoUnLF + rhoUnRF)
    rhoUFlux = 0.5*(rhoUnLF*ULF + rhoUnRF*URF + (pLF + pRF)*Normals)
    rhoEFlux = 0.5*(rhoUnLF*hLF + rhoUnRF*hRF)

    sqrtRhoLF, sqrtRhoRF = rhoLF.sqrt(), rhoRF.sqrt()
    divRhoF = sqrtRhoLF + sqrtRhoRF
    UF = (ULF*sqrtRhoLF + URF*sqrtRhoRF)/divRhoF
    hF = (hLF*sqrtRhoLF + hRF*sqrtRhoRF)/divRhoF

    qF = 0.5*UF.magSqr()
    a2F = (gamma-1)*(hF-qF)
    # speed of sound for CFL
    aF = a2F.sqrt()
    # normal velocity for CFL
    UnF = UF.dot(Normals)

    drhoF = rhoRF - rhoLF
    drhoUF = rhoRF*URF - rhoLF*ULF
    drhoEF = (hRF*rhoRF-pRF)-(hLF*rhoLF-pLF)

    lam1, lam2, lam3 = UnF.abs(), (UnF + aF).abs(), (UnF - aF).abs()

    eps = 0.5*(rhoUnLF/rhoLF - rhoUnRF/rhoRF).abs()
    eps += 0.5*((gamma*pLF/rhoLF).sqrt() - (gamma*pRF/rhoRF).sqrt()).abs()
    eps = eps.stabilise(config.SMALL)

    #lam1 = Field.switch(ad.(lam1.field, 2.*eps.field), 0.25*lam1*lam1/eps + eps, lam1)
    lam1 = Tensor.switch(lam1.scalars[0] < 2.*eps.scalars[0], .25*lam1*lam1/eps + eps, lam1)
    lam2 = Tensor.switch(lam2.scalars[0] < 2.*eps.scalars[0], .25*lam2*lam2/eps + eps, lam2)
    lam3 = Tensor.switch(lam3.scalars[0] < 2.*eps.scalars[0], .25*lam3*lam3/eps + eps, lam3)

    abv1 = 0.5*(lam2 + lam3)
    abv2 = 0.5*(lam2 - lam3)
    abv3 = abv1 - lam1
    abv4 = (gamma-1)*(qF*drhoF - UF.dot(drhoUF) + drhoEF)
    abv5 = UnF*drhoF - drhoUF.dot(Normals)
    abv6 = abv3*abv4/a2F - abv2*abv5/aF
    abv7 = abv3*abv5 - abv2*abv4/aF

    rhoFlux -= 0.5*(lam1*drhoF + abv6)
    rhoUFlux -= 0.5*(lam1*drhoUF + UF*abv6 - abv7*Normals)
    rhoEFlux -= 0.5*(lam1*drhoEF + hF*abv6 - UnF*abv7)

    return rhoFlux, rhoUFlux, rhoEFlux


def eulerHLLC(gamma, pLF, pRF, TLF, TRF, ULF, URF, \
                rhoLF, rhoRF, rhoULF, rhoURF, rhoELF, rhoERF, Normals):

    UnLF, UnRF = ULF.dot(Normals), URF.dot(Normals)
    qLF, qRF = ULF.magSqr(), URF.magSqr()
    cLF, cRF = (gamma*pLF)/rhoLF, (gamma*pRF)/rhoRF
    hLF = gamma*pLF/((gamma-1)*rhoLF) + 0.5*qLF
    hRF = gamma*pRF/((gamma-1)*rhoRF) + 0.5*qRF
    eLF, eRF = hLF*rhoLF-pLF, hRF*rhoRF-pRF

    RrhoF = (rhoRF/rhoLF).sqrt()
    divRhoF = RrhoF + 1
    UF = (ULF + URF*RrhoF)/divRhoF
    # normal velocity for CFL
    UnF = UF.dot(Normals)

    PrhoF = (cLF.sqr()+0.5*(gamma-1)*qLF + (cRF.sqr()+0.5*(gamma-1)*qRF)*RrhoF)/divRhoF
    cF = (PrhoF-0.5*(gamma-1)*UF.magSqr()).sqrt()
    # speed of sound for CFL
    #aF = cF

    sLF = Field.min(UnF-cF, UnLF-cLF)
    sRF = Field.max(UnF+cF, UnRF+cRF)

    sMF = (pLF-pRF - rhoLF*UnLF*(sLF-UnLF) + rhoRF*UnRF*(sRF-UnRF)) \
          /(rhoRF*(sRF-UnRF)-rhoLF*(sLF-UnRF))
    pSF = rhoRF*(UnRF-sRF)*(UnRF-sMF) + pRF

    Frho1 = rhoLF*UnLF
    FrhoU1 = Frho1*ULF + pLF*Normals
    FrhoE1 = (eLF + pLF)*UnLF

    divsLF = sLF-sMF
    sUnLF = sLF-UnLF
    rhosLF = rhoLF*sUnLF/divsLF
    rhoUsLF = (rhoLF*ULF*sUnLF + (pSF-pLF)*Normals)/divsLF
    esLF = (sUnLF*eLF-pLF*UnLF+pSF*sMF)/divsLF

    Frho2 = rhosLF*sMF
    FrhoU2 = rhoUsLF*sMF + pSF*Normals
    FrhoE2 = (esLF + pSF)*sMF

    divsRF = sRF-sMF
    sUnRF = sRF-UnRF
    rhosRF = rhoRF*sUnRF/divsRF
    rhoUsRF = (rhoRF*URF*sUnRF + (pSF-pRF)*Normals)/divsRF
    esRF = (sUnRF*eRF-pRF*UnRF+pSF*sMF)/divsRF

    Frho3 = rhosRF*sMF
    FrhoU3 = rhoUsRF*sMF + pSF*Normals
    FrhoE3 = (esRF + pSF)*sMF

    Frho4 = rhoRF*UnRF
    FrhoU4 = Frho1*URF + pRF*Normals
    FrhoE4 = (eRF + pRF)*UnRF

    rhoFlux = Field.switch(ad.gt(sMF.field, 0.), \
              Field.switch(ad.gt(sLF.field, 0.), Frho1, Frho2), \
              Field.switch(ad.gt(sRF.field, 0.), Frho3, Frho4))
    
    rhoUFlux = Field.switch(ad.gt(sMF.field, 0.), \
              Field.switch(ad.gt(sLF.field, 0.), FrhoU1, FrhoU2), \
              Field.switch(ad.gt(sRF.field, 0.), FrhoU3, FrhoU4))

    rhoEFlux = Field.switch(ad.gt(sMF.field, 0.), \
              Field.switch(ad.gt(sLF.field, 0.), FrhoE1, FrhoE2), \
              Field.switch(ad.gt(sRF.field, 0.), FrhoE3, FrhoE4))

    return rhoFlux, rhoUFlux, rhoEFlux
