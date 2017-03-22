#ifndef OBJECTIVE_HPP
#define OBJECTIVE_HPP

#include "common.hpp"
#include "density.hpp"

scalar objectiveNone(RCF* rcf, const arr& U, const arr& T, const arr& p);
scalar objectiveDrag(RCF* rcf, const arr& U, const arr& T, const arr& p);
#endif
