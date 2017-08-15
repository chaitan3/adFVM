#ifndef PARALLEL_HPP
#define PARALLEL_HPP

#include "interface.hpp"

template <typename dtype, integer shape1, integer shape2>
void Function_mpi_init(std::vector<extArrType<dtype, shape1, shape2>*> phiP);
template <typename dtype, integer shape1, integer shape2>
void Function_mpi(std::vector<extArrType<dtype, shape1, shape2>*> phiP);
template <typename dtype, integer shape1, integer shape2>
void Function_mpi_end(std::vector<extArrType<dtype, shape1, shape2>*> phiP);
void Function_mpi_allreduce(std::vector<ext_vec*> vals);

template <typename dtype, integer shape1, integer shape2>
void Function_mpi_init_grad(std::vector<extArrType<dtype, shape1, shape2>*> phiP);
template <typename dtype, integer shape1, integer shape2>
void Function_mpi_grad(std::vector<extArrType<dtype, shape1, shape2>*> phiP);
template <typename dtype, integer shape1, integer shape2>
void Function_mpi_end_grad(std::vector<extArrType<dtype, shape1, shape2>*> phiP);
void Function_mpi_allreduce_grad(std::vector<ext_vec*> vals);


template<typename T> inline MPI_Datatype mpi_type();
template<> inline MPI_Datatype mpi_type<float>() {return MPI_FLOAT;}
template<> inline MPI_Datatype mpi_type<double>() {return MPI_DOUBLE;}


#endif
