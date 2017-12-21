from . import config
from .field import Field

from adpy.tensor import Tensor

def eulerLaxFriedrichs(gamma, pLF, pRF, TLF, TRF, ULF, URF, \
                rhoLF, rhoRF, rhoULF, rhoURF, rhoELF, rhoERF, Normals):

    UnLF, UnRF = ULF.dot(Normals), URF.dot(Normals)
    cLF = (gamma*pLF/rhoLF).sqrt()
    cRF = (gamma*pRF/rhoRF).sqrt()
    aF = Tensor.max(abs(UnLF) + cLF, abs(UnRF) + cRF)

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

    lam1, lam2, lam3 = abs(UnF), abs(UnF + aF), abs(UnF - aF)

    eps = 0.5*abs(rhoUnLF/rhoLF - rhoUnRF/rhoRF)
    eps += 0.5*abs((gamma*pLF/rhoLF).sqrt() - (gamma*pRF/rhoRF).sqrt())
    eps = eps.stabilise(config.SMALL)

    lam1 = Tensor.switch(lam1 < 2.*eps, .25*lam1*lam1/eps + eps, lam1)
    lam2 = Tensor.switch(lam2 < 2.*eps, .25*lam2*lam2/eps + eps, lam2)
    lam3 = Tensor.switch(lam3 < 2.*eps, .25*lam3*lam3/eps + eps, lam3)

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

