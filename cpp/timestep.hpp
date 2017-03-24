#include "density.hpp"

tuple<scalar,scalar> euler(RCF *rcf, const arr& rho, const arr& rhoU, const arr& rhoE, arr& rhoN, arr& rhoUN, arr& rhoEN, scalar t, scalar dt);
tuple<scalar,scalar> SSPRK(RCF *rcf, const arr& rho, const arr& rhoU, const arr& rhoE, arr& rhoN, arr& rhoUN, arr& rhoEN, scalar t, scalar dt);
