#include "density.hpp"

scalar euler(RCF *rcf, const arr& rho, const arr& rhoU, const arr& rhoE, arr& rhoN, arr& rhoUN, arr& rhoEN, scalar t, scalar dt);
scalar SSPRK(RCF *rcf, const arr& rho, const arr& rhoU, const arr& rhoE, arr& rhoN, arr& rhoUN, arr& rhoEN, scalar t, scalar dt);
