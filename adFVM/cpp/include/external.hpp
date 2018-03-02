#ifndef EXTERNAL_HPP
#define EXTERNAL_HPP

#include "mesh.hpp"
#include "parallel.hpp"
#include "scaling.hpp"

void external_init(PyObject*);
void external_exit();

extern Mesh *meshp;
#ifdef GPU
    extern cusolverDnHandle_t cusolver_handle;
    extern cublasHandle_t cublas_handle;
#endif

#endif
