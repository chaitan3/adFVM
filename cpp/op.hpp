#ifndef OP_HPP
#define OP_HPP

#include "interface.hpp"

class Operator {
    Mesh const * mesh;
    
    public:
        Operator(Mesh const* mesh): mesh(mesh) {};
        template<integer shape1, integer shape2>
        void div(const scalar*, arrType<scalar, shape1, shape2>&, integer, bool);
        template<integer shape1, integer shape2>
        void absDiv(const scalar*, arrType<scalar, shape1, shape2>&, integer, bool);
        template<integer shape1, integer shape2>
        void grad(const scalar*, arrType<scalar, shape1, shape2>&, integer, bool);
        template<integer shape1, integer shape2>
        void snGrad(const arrType<scalar, shape1, shape2>&, scalar *, integer);
        //void laplacian(const arr&, const arr&, arr&);
};

 
#endif
