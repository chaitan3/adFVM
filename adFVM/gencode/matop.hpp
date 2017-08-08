#include "common.hpp"

#ifndef MATOP_HPP
#define MATOP_HPP

#include <petscksp.h>

class Matop {
    map<string, ivec> boundaryNeighbours;
    map<string, integer> boundaryProcs;
    scalar norm = -1;

    public:

    Matop(RCF* rcf);    
    void heat_equation(RCF *rcf, const arrType<scalar, 5>& u, const vec& DT, const scalar dt, arrType<scalar, 5>& un);
    void viscosity(const vec& rho, const mat& rhoU, const vec& rhoE, vec& M_2norm, vec& DT, scalar scaling, bool report);
};
#endif
