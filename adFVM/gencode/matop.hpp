#include "common.hpp"

#ifndef MATOP_HPP
#define MATOP_HPP

#include <petscksp.h>

class Matop {
    map<string, ivec> boundaryNeighbours;
    map<string, integer> boundaryProcs;
    scalar norm = -1;

    public:

    Matop();    
    void heat_equation(const arrType<scalar, 5>& u, const vec& DT, const scalar dt, arrType<scalar, 5>& un);
};
#endif
