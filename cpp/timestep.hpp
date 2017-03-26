#include "density.hpp"

tuple<scalar, scalar> euler(RCF *rcf, const vec& rho, const mat& rhoU, const vec& rhoE, vec& rhoN, mat& rhoUN, vec& rhoEN, scalar t, scalar dt);
tuple<scalar, scalar> SSPRK(RCF *rcf, const vec& rho, const mat& rhoU, const vec& rhoE, vec& rhoN, mat& rhoUN, vec& rhoEN, scalar t, scalar dt);
