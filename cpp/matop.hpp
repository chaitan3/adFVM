#include "common.hpp"
#include "density.hpp"
#include <petscksp.h>

class Matop {
    map<string, ivec> boundaryNeighbours;
    map<string, integer> boundaryProcs;

    public:

    Matop(RCF* rcf);    
    void heat_equation(RCF *rcf, const arrType<uscalar, 5> u, const uvec DT, const uscalar dt, arrType<uscalar, 5>& un);
};
