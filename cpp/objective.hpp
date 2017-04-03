#ifndef OBJECTIVE_HPP
#define OBJECTIVE_HPP

#include "common.hpp"
#include "density.hpp"

scalar objectiveNone(RCF* rcf, const mat& U, const vec& T, const vec& p);
scalar objectiveDrag(RCF* rcf, const mat& U, const vec& T, const vec& p);
scalar objectivePressureLoss(RCF* rcf, const mat& U, const vec& T, const vec& p);
#endif
