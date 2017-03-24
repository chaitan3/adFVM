#include "interface.hpp"
#include "op.hpp"
#include "interp.hpp"

#ifndef DENSITY_HPP
#define DENSITY_HPP

//#define faceReconstructor secondOrder
#define faceReconstructor firstOrder
#define riemannSolver eulerRoe
//#define riemannSolver eulerLaxFriedrichs
#define boundaryRiemannSolver eulerRoe
//#define boundaryRiemannSolver eulerLaxFriedrichs

class RCF {
    
    public:

    scalar R = 8.314;
    scalar Cp = 1004.5;
    scalar gamma = 1.4;
    scalar Cv = Cp/gamma;
    scalar Pr = 0.7;
    scalar CFL = 1.2;
    scalar stepFactor = 1.2;

    scalar muC;
    scalar (RCF::*mu)(const scalar);

    scalar sutherland(const scalar T) {
      return 1.4792e-06*pow(T, 1.5)/(T+116);
    }
    scalar constantMu(const scalar T) {
      return this->muC;
    }
    scalar kappa(const scalar mu, const scalar T) {
      return mu*this->Cp/this->Pr;
    }
    

    arr* U;
    arr* T;
    arr* p;
    Boundary* boundaries;

    Interpolator* interpolate;
    Operator* operate;
    Mesh const* mesh;
    scalar (*objective)(RCF*, const arr&, const arr&, const arr&);
    string objectiveDragInfo;

    void primitive(const scalar rho, const scalar rhoU[3], const scalar rhoE, scalar U[3], scalar& T, scalar& p);
    void conservative(const scalar U[3], const scalar T, const scalar p, scalar& rho, scalar rhoU[3], scalar& rhoE);
    void getFlux(const scalar U[3], const scalar T, const scalar p, const uscalar N[3], scalar& rhoFlux, scalar rhoUFlux[3], scalar& rhoEFlux);

    void equation(const arr& rho, const arr& rhoU, const arr& rhoE, arr& drho, arr& drhoU, arr& drhoE, scalar& objective);
    template<typename dtype>
    void boundary(const Boundary& boundary, arrType<dtype>& phi);

    void setMesh(Mesh const* mesh)  {
        this->mesh = mesh;
        this->interpolate = new Interpolator(mesh);
        this->operate = new Operator(mesh);
        this->boundaries = new Boundary[3];
        this->mu = &RCF::sutherland;
    }

    ~RCF() {
        delete this->interpolate;
        delete this->operate;
        delete[] this->boundaries;
    }
};

#endif
