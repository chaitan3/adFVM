#ifndef INTERP_HPP
#define INTERP_HPP

#include "field.hpp"

class Interpolator {
    const Mesh& mesh;
    
    Interpolator(const Mesh& mesh): mesh(mesh) {};
    mat central(const mat&);
    mat TVD(const mat&);
};
 

#endif
