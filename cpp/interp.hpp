#ifndef INTERP_HPP
#define INTERP_HPP

#include "interface.hpp"

class Interpolator {
    Mesh const* mesh;
    
    public:
        Interpolator(Mesh const* mesh): mesh(mesh) {};
        template<integer shape1, integer shape2>
        void central(const arrType<scalar, shape1, shape2>& phi, scalar* phiF, integer index);
        template<integer shape1, integer shape2>
        void average(const arrType<scalar, shape1, shape2>& phi, scalar* phiF, integer index);
        template<integer shape1>
        void firstOrder(const arrType<scalar, shape1>& phi, const arrType<scalar, shape1, 3>& gradPhi, scalar *phiF, integer index, integer swap);
        template<integer shape1>
        void secondOrder(const arrType<scalar, shape1>& phi, const arrType<scalar, shape1, 3>& gradPhi, scalar *phiF, integer index, integer swap);
};
 

#endif
