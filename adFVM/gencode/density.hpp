#ifndef DENSITY_HPP
#define DENSITY_HPP

//#define timeIntegrator euler
//#define timeIntegrator_grad euler_grad
//#define nStages 1

#include "common.hpp"
#include "interface.hpp"

class RCF {
    public:

    void* req;
    integer reqIndex;
    integer reqField;
    Boundary boundaries[3];
    scalar* reqBuf[6];
    integer stage;
    scalar CFL;
    map<string, string> objectivePLInfo;

    void equation(const vec& rho, const mat& rhoU, const vec& rhoE, vec& drho, mat& drhoU, vec& drhoE, scalar& obj, scalar& minDtc);
    void boundaryUPT(mat& U, vec& T, vec& p);
    void boundaryInit(integer startField);
    template <typename dtype, integer shape1, integer shape2>
    void boundary(const Boundary& boundary, arrType<dtype, shape1, shape2>& phi);
    void boundaryEnd();

    void equation_grad(const vec& rho, const mat& rhoU, const vec& rhoE, const vec& drhoa, const mat& drhoUa, const vec& drhoEa, vec& rhoa, mat& rhoUa, vec& rhoEa, scalar& obj, scalar& minDtc);
    template <typename dtype, integer shape1, integer shape2>
    void boundary_grad(const Boundary& boundary, arrType<dtype, shape1, shape2>& phi);
    template <typename dtype, integer shape1, integer shape2>
    void boundaryEnd_grad(arrType<dtype, shape1, shape2>& phi, dtype* phiBuf);
    void boundaryUPT_grad(const mat& U, const vec& T, const vec& p, mat& Ua, vec& Ta, vec& pa);

};

void timeIntegrator_init(const vec& rho, const mat& rhoU, const vec& rhoE, vec& rhoN, mat& rhoUN, vec& rhoEN);
void timeIntegrator_exit();
tuple<scalar, scalar> euler(const vec& rho, const mat& rhoU, const vec& rhoE, vec& rhoN, mat& rhoUN, vec& rhoEN, scalar t, scalar dt);
tuple<scalar, scalar> SSPRK(const vec& rho, const mat& rhoU, const vec& rhoE, vec& rhoN, mat& rhoUN, vec& rhoEN, scalar t, scalar dt);
tuple<scalar, scalar> euler_grad(const vec& rho, const mat& rhoU, const vec& rhoE, const vec& rhoa, const mat& rhoUa, const vec& rhoEa, vec& rhoaN, mat& rhoUaN, vec& rhoEaN, scalar t, scalar dt);
tuple<scalar, scalar> SSPRK_grad(const vec& rho, const mat& rhoU, const vec& rhoE, const vec& rhoa, const mat& rhoUa, const vec& rhoEa, vec& rhoaN, mat& rhoUaN, vec& rhoEaN, scalar t, scalar dt);


#define timeIntegrator SSPRK
#define timeIntegrator_grad SSPRK_grad
#define nStages 3


extern RCF* rcf;
extern vec *rhos[nStages+1];
extern mat *rhoUs[nStages+1];
extern vec *rhoEs[nStages+1];
extern mat *Us[nStages];
extern vec *Ts[nStages];
extern vec *ps[nStages];
extern arrType<scalar, 3, 3> *gradUs[nStages];
extern arrType<scalar, 1, 3> *gradTs[nStages];
extern arrType<scalar, 1, 3> *gradps[nStages];


#endif
