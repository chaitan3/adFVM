#ifndef INTERP_HPP
#define INTERP_HPP

#include "interface.hpp"

class Interpolator {
    const Mesh& mesh;
    
    public:
        Interpolator(const Mesh& mesh): mesh(mesh) {};
        arr central(const arr&);
        arr TVD(const arr&, const arr&, const arr& );
};
 

#endif
