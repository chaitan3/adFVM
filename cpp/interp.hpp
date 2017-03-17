#ifndef INTERP_HPP
#define INTERP_HPP

#include "interface.hpp"

class Interpolator {
    const Mesh& mesh;
    
    public:
        Interpolator(const Mesh& mesh): mesh(mesh) {};
        void central(const arr&, scalar [], integer);
        void secondOrder(const arr& phi, const arr& gradPhi, scalar *phiF, integer index, integer which);
};
 

#endif
