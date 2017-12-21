#ifndef MATOP_HPP
#define MATOP_HPP

#include <petscksp.h>
#include <petscvec.h>
#ifdef GPU
    #include <petsccuda.h>
#endif
#include "mesh.hpp"

class Matop {
    map<string, ivec> boundaryNeighbours;
    map<string, integer> boundaryProcs;
    scalar norm = -1;

    public:

    Matop();    
    ~Matop();    
    int heat_equation(vector<ext_vec*> u, const ext_vec& DT, const ext_vec& dt, vector<ext_vec*> un);
};

extern Matop *matop;
#endif
