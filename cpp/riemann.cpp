#include "riemann.hpp"

void eulerLaxFriedrichs(
                    const scalar gamma, 
                    const scalar pLF, 
                    const scalar pRF, 
                    const scalar TLF, 
                    const scalar TRF, 
                    const scalar ULF[3], 
                    const scalar URF[3], 
                    const scalar rhoLF, 
                    const scalar rhoRF, 
                    const scalar rhoULF[3], 
                    const scalar rhoURF[3], 
                    const scalar rhoELF, 
                    const scalar rhoERF, 
                    const scalar Normals[3],
                    scalar& rhoFlux,
                    scalar rhoUFlux[3],
                    scalar& rhoEFlux) {

    scalar UnLF = 0., UnRF = 0.;
    for (integer i = 0; i < 3; i++) {
        UnLF += ULF[i]*Normals[i];
        UnRF += URF[i]*Normals[i];
    }
    scalar cLF = sqrt(gamma*pLF/rhoLF);
    scalar cRF = sqrt(gamma*pRF/rhoRF);
    scalar aF = max(fabs(UnLF) + cLF, fabs(UnRF) + cRF);

    rhoFlux = 0.5*(rhoLF*UnLF + rhoRF*UnRF) - 0.5*aF*(rhoRF-rhoLF);
    for (integer i = 0; i < 3; i++) {
        rhoUFlux[i] = 0.5*(rhoULF[i]*UnLF + rhoURF[i]*UnRF + (pLF + pRF)*Normals[i]) - 0.5*aF*(rhoURF[i]-rhoULF[i]);
    }
    rhoEFlux = 0.5*((rhoELF + pLF)*UnLF + (rhoERF + pRF)*UnRF) - 0.5*aF*(rhoERF-rhoELF);
}

void eulerRoe(
                    const scalar gamma, 
                    const scalar pLF, 
                    const scalar pRF, 
                    const scalar TLF, 
                    const scalar TRF, 
                    const scalar ULF[3], 
                    const scalar URF[3], 
                    const scalar rhoLF, 
                    const scalar rhoRF, 
                    const scalar rhoULF[3], 
                    const scalar rhoURF[3], 
                    const scalar rhoELF, 
                    const scalar rhoERF, 
                    const uscalar Normals[3],
                    scalar& rhoFlux,
                    scalar rhoUFlux[3],
                    scalar& rhoEFlux) {

    scalar rhoUnLF = 0.;
    scalar rhoUnRF = 0.;
    scalar ULF2 = 0.;
    scalar URF2 = 0.;
    for (integer i = 0; i < 3; i++) {
        rhoUnLF += rhoLF*ULF[i]*Normals[i];
        rhoUnRF += rhoRF*URF[i]*Normals[i];
        ULF2 += ULF[i]*ULF[i];
        URF2 += URF[i]*URF[i];
    }
    scalar hLF = gamma*pLF/((gamma-1)*rhoLF) + 0.5*ULF2;
    scalar hRF = gamma*pRF/((gamma-1)*rhoRF) + 0.5*URF2;

    rhoFlux = 0.5*(rhoUnLF + rhoUnRF);
    for (integer i = 0; i < 3; i++) {
        rhoUFlux[i] = 0.5*(rhoUnLF*ULF[i] + rhoUnRF*URF[i] + (pLF + pRF)*Normals[i]);
    }
    rhoEFlux = 0.5*(rhoUnLF*hLF + rhoUnRF*hRF);

    scalar sqrtRhoLF = sqrt(rhoLF);
    scalar sqrtRhoRF = sqrt(rhoRF);
    scalar divRhoF = sqrtRhoLF + sqrtRhoRF;
    scalar UF[3], qF = 0., UnF = 0.;
    for (integer i = 0; i < 3; i++) {
        UF[i] = (ULF[i]*sqrtRhoLF + URF[i]*sqrtRhoRF)/divRhoF;
        qF += 0.5*UF[i]*UF[i];
        UnF += UF[i]*Normals[i];
    }
    scalar hF = (hLF*sqrtRhoLF + hRF*sqrtRhoRF)/divRhoF;

    scalar a2F = (gamma-1)*(hF-qF);
    //speed of sound for CFL
    scalar aF = sqrt(a2F);
    //normal velocity for CFL

    scalar drhoF = rhoRF - rhoLF;
    scalar drhoUF[3];
    for (integer i = 0; i < 3; i++) {
        drhoUF[i] = rhoRF*URF[i] - rhoLF*ULF[i];
    }
    scalar drhoEF = (hRF*rhoRF-pRF)-(hLF*rhoLF-pLF);

    scalar lam1 = fabs(UnF);
    scalar lam2 = fabs(UnF + aF);
    scalar lam3 = fabs(UnF - aF);

    scalar eps = fabs(0.5*(rhoUnLF/rhoLF - rhoUnRF/rhoRF));
    eps += 0.5*fabs(sqrt(gamma*pLF/rhoLF) - sqrt(gamma*pRF/rhoRF));
    
    #ifdef ADIFF
        if (eps.value() < 0) 
            eps -= SMALL;
        else
            eps += SMALL;
        if (lam1.value() < 2*eps) lam1 = .25*lam1*lam1/eps;
        if (lam2.value() < 2*eps) lam2 = .25*lam2*lam2/eps;
        if (lam3.value() < 2*eps) lam3 = .25*lam3*lam3/eps;
    #else
        eps = eps < 0 ? eps-SMALL : eps + SMALL;
        lam1 = lam1 < 2*eps ? .25*lam1*lam1/eps + eps : lam1;
        lam2 = lam2 < 2*eps ? .25*lam2*lam3/eps + eps : lam2;
        lam3 = lam3 < 2*eps ? .25*lam3*lam3/eps + eps : lam3;
    #endif

    scalar abv1 = 0.5*(lam2 + lam3);
    scalar abv2 = 0.5*(lam2 - lam3);
    scalar abv3 = abv1 - lam1;
    scalar tmp = 0., tmp2 = 0.;
    for (integer i = 0; i < 3; i++) {
        tmp += UF[i]*drhoUF[i];
        tmp2 += drhoUF[i]*Normals[i];
    }
    scalar abv4 = (gamma-1)*(qF*drhoF - tmp + drhoEF);
    scalar abv5 = UnF*drhoF - tmp2;
    scalar abv6 = abv3*abv4/a2F - abv2*abv5/aF;
    scalar abv7 = abv3*abv5 - abv2*abv4/aF;

    rhoFlux -= 0.5*(lam1*drhoF + abv6);
    for (integer i = 0; i < 3; i++) {
        rhoUFlux[i] -= 0.5*(lam1*drhoUF[i] + UF[i]*abv6 - abv7*Normals[i]);
    }
    rhoEFlux -= 0.5*(lam1*drhoEF + hF*abv6 - UnF*abv7);

}


//def eulerHLLC(gamma, pLF, pRF, TLF, TRF, ULF, URF, \
//                rhoLF, rhoRF, rhoULF, rhoURF, rhoELF, rhoERF, Normals):
//
//    UnLF, UnRF = ULF.dot(Normals), URF.dot(Normals)
//    qLF, qRF = ULF.magSqr(), URF.magSqr()
//    cLF, cRF = (gamma*pLF)/rhoLF, (gamma*pRF)/rhoRF
//    hLF = gamma*pLF/((gamma-1)*rhoLF) + 0.5*qLF
//    hRF = gamma*pRF/((gamma-1)*rhoRF) + 0.5*qRF
//    eLF, eRF = hLF*rhoLF-pLF, hRF*rhoRF-pRF
//
//    RrhoF = (rhoRF/rhoLF).sqrt()
//    divRhoF = RrhoF + 1
//    UF = (ULF + URF*RrhoF)/divRhoF
//    # normal velocity for CFL
//    UnF = UF.dot(Normals)
//
//    PrhoF = (cLF.sqr()+0.5*(gamma-1)*qLF + (cRF.sqr()+0.5*(gamma-1)*qRF)*RrhoF)/divRhoF
//    cF = (PrhoF-0.5*(gamma-1)*UF.magSqr()).sqrt()
//    # speed of sound for CFL
//    #aF = cF
//
//    sLF = Field.min(UnF-cF, UnLF-cLF)
//    sRF = Field.max(UnF+cF, UnRF+cRF)
//
//    sMF = (pLF-pRF - rhoLF*UnLF*(sLF-UnLF) + rhoRF*UnRF*(sRF-UnRF)) \
//          /(rhoRF*(sRF-UnRF)-rhoLF*(sLF-UnRF))
//    pSF = rhoRF*(UnRF-sRF)*(UnRF-sMF) + pRF
//
//    Frho1 = rhoLF*UnLF
//    FrhoU1 = Frho1*ULF + pLF*Normals
//    FrhoE1 = (eLF + pLF)*UnLF
//
//    divsLF = sLF-sMF
//    sUnLF = sLF-UnLF
//    rhosLF = rhoLF*sUnLF/divsLF
//    rhoUsLF = (rhoLF*ULF*sUnLF + (pSF-pLF)*Normals)/divsLF
//    esLF = (sUnLF*eLF-pLF*UnLF+pSF*sMF)/divsLF
//
//    Frho2 = rhosLF*sMF
//    FrhoU2 = rhoUsLF*sMF + pSF*Normals
//    FrhoE2 = (esLF + pSF)*sMF
//
//    divsRF = sRF-sMF
//    sUnRF = sRF-UnRF
//    rhosRF = rhoRF*sUnRF/divsRF
//    rhoUsRF = (rhoRF*URF*sUnRF + (pSF-pRF)*Normals)/divsRF
//    esRF = (sUnRF*eRF-pRF*UnRF+pSF*sMF)/divsRF
//
//    Frho3 = rhosRF*sMF
//    FrhoU3 = rhoUsRF*sMF + pSF*Normals
//    FrhoE3 = (esRF + pSF)*sMF
//
//    Frho4 = rhoRF*UnRF
//    FrhoU4 = Frho1*URF + pRF*Normals
//    FrhoE4 = (eRF + pRF)*UnRF
//
//    rhoFlux = Field.switch(ad.gt(sMF.field, 0.), \
//              Field.switch(ad.gt(sLF.field, 0.), Frho1, Frho2), \
//              Field.switch(ad.gt(sRF.field, 0.), Frho3, Frho4))
//    
//    rhoUFlux = Field.switch(ad.gt(sMF.field, 0.), \
//              Field.switch(ad.gt(sLF.field, 0.), FrhoU1, FrhoU2), \
//              Field.switch(ad.gt(sRF.field, 0.), FrhoU3, FrhoU4))
//
//    rhoEFlux = Field.switch(ad.gt(sMF.field, 0.), \
//              Field.switch(ad.gt(sLF.field, 0.), FrhoE1, FrhoE2), \
//              Field.switch(ad.gt(sRF.field, 0.), FrhoE3, FrhoE4))
//
//    r

