#ifndef MATOP_HPP
#define MATOP_HPP

#include <petscksp.h>
#include "common.hpp"
#include "mesh.hpp"

class Matop {
    map<string, ivec> boundaryNeighbours;
    map<string, integer> boundaryProcs;
    scalar norm = -1;

    public:

    Matop();    
    ~Matop();    
    int heat_equation(vector<vec*> u, const vec& DT, const scalar dt, vector<vec*> un);
};

extern Matop *matop;
#endif
