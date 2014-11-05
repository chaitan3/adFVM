#ifndef OP_HPP
#define OP_HPP

#include "interface.hpp"

class Operator {
    const Mesh& mesh;
    arr internal_sum(const arr&);
    
    public:
        Operator(const Mesh& mesh): mesh(mesh) {};
        arr div(const arr&);
        arr grad(const arr&);
        arr snGrad(const arr&);
        arr laplacian(const arr&, const arr&);
};

 
#endif
