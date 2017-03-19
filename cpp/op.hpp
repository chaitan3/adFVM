#ifndef OP_HPP
#define OP_HPP

#include "interface.hpp"

class Operator {
    Mesh const * mesh;
    arr internal_sum(const arr&);
    
    public:
        Operator(Mesh const* mesh): mesh(mesh) {};
        void div(const scalar*, arr&, integer, bool);
        void grad(const scalar*, arr&, integer, bool);
        //void snGrad(const arr&, arr&);
        //void laplacian(const arr&, const arr&, arr&);
};

 
#endif
