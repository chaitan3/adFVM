#ifndef INTERP_HPP
#define INTERP_HPP

#include "field.hpp"

class Interpolator {
    const Mesh& mesh;
    
    public:
        Interpolator(const Mesh& mesh): mesh(mesh) {};
        arr central(const arr&);
        arr TVD(const arr&);
};
 

#endif
