#include "common.hpp"

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
                    scalar& rhoEFlux);

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
                    const scalar Normals[3],
                    scalar& rhoFlux,
                    scalar rhoUFlux[3],
                    scalar& rhoEFlux);


