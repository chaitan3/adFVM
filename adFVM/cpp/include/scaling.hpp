#ifndef SCALING_HPP
#define SCALING_HPP

#include "interface.hpp"
#include "mesh.hpp"

void Function_get_max_eigenvalue(std::vector<extArrType<scalar, 5, 5>*> phiP);
void Function_get_max_generalized_eigenvalue(vector<extArrType<scalar, 5, 5>*> phiP);
void Function_apply_adjoint_viscosity(std::vector<extArrType<scalar, 1>*> phiP);

#endif
