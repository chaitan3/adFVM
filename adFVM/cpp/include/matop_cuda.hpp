#ifndef MATOP_HPP
#define MATOP_HPP

#include<cusparse.h>
#include<cusolverSp.h>

#include "mesh.hpp"

class Matop {
    cusolverSpHandle_t handle = NULL;
    cusparseHandle_t cusparseHandle = NULL;
    extArrType<int, 1> indptr;
    extArrType<int, 1> indices;
    extArrType<scalar, 1> volumes;
    extArrType<scalar, 1> data;
    extArrType<scalar, 6> cellFaces;
    int nnz;

    public:

    Matop();    
    ~Matop();    
    int heat_equation(vector<ext_vec*> u, const ext_vec& DT, const ext_vec& dt, vector<ext_vec*> un);
};

extern Matop *matop;
#endif
