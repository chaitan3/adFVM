#ifndef OP_HPP
#define OP_HPP

#include "interface.hpp"

class Operator {
    Mesh const * mesh;
    
    public:
        Operator(Mesh const* mesh): mesh(mesh) {};
        template<integer shape1>
        void div(const scalar*, arrType<scalar, shape1>&, integer, bool);
        template<integer shape1>
        void absDiv(const scalar*, arrType<scalar, shape1>&, integer, bool);
        template<integer shape1>
        void grad(const scalar*, arrType<scalar, shape1, 3>&, integer, bool);
        template<integer shape1>
        void snGrad(const arrType<scalar, shape1>&, scalar *, integer);
        //void laplacian(const arr&, const arr&, arr&);
};

 
#endif
