#ifndef INTERP_HPP
#define INTERP_HPP

#include "interface.hpp"

class Interpolator {
    Mesh const* mesh;
    
    public:
        Interpolator(Mesh const* mesh): mesh(mesh) {};
        void central(const arr&, scalar [], integer);
        void average(const arr&, scalar [], integer);
        void firstOrder(const arr& phi, const arr& gradPhi, scalar *phiF, integer index, integer which);
        void secondOrder(const arr& phi, const arr& gradPhi, scalar *phiF, integer index, integer which);
};
 

#endif
