#ifndef OP_HPP
#define OP_HPP

#include "interface.hpp"

class Operator {
    const Mesh& mesh;
    arr internal_sum(const arr&);
    
    public:
        Operator(const Mesh& mesh): mesh(mesh) {};
        void div(const scalar*, arr&, integer);
        void grad(const scalar*, arr&, integer);
        //void snGrad(const arr&, arr&);
        //void laplacian(const arr&, const arr&, arr&);
};

 
#endif
