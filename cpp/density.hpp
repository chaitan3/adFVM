#include "interface.hpp"
#include "op.hpp"
#include "interp.hpp"

#ifndef DENSITY_HPP
#define DENSITY_HPP

class RCF {
    
    scalar R = 8.314;
    //double Cp = 1004.5;
    scalar Cp = 2.5;
    scalar gamma = 1.4;
    scalar Cv = Cp/gamma;
    scalar Pr = 0.7;
    scalar CFL = 1.2;
    scalar stepFactor = 1.2;

    Interpolator* interpolate;
    Operator* operate;
    arr* U;
    arr* T;
    arr* p;

    //// confirm that make_tuple doesn't create copies

    //inline Ref<arr> internalField(arr& phi) {
        //Ref<arr> phiI = SELECT(phi, 0, mesh.nInternalCells);
        //return phiI;
    //}

    //inline Ref<arr> boundaryField(arr& phi) { 
        //Ref<arr> phiB = SELECT(phi, mesh.nInternalCells, mesh.nGhostCells);
        //return phiB;
    //}
    
    //inline arr mu(const arr& T) {
        //return 0*T;//1.4792e-06*T.pow(1.5)/(T+116);
    //}
    //inline arr kappa(const arr& mu, const arr& T) {
        //return mu*this->Cp/this->Pr;
    //}
    
    //public:
        //RCF (string caseDir, double time):
            //mesh(caseDir), 
            //U("U", mesh, time), T("T", mesh, time), p("p", mesh, time),
            //interpolate(mesh), operate(mesh) {};

    public:
    Mesh const* mesh;
    Boundary* boundaries;
    void primitive(const scalar rho, const scalar rhoU[3], const scalar rhoE, scalar U[3], scalar& T, scalar& p);
    void conservative(const scalar U[3], const scalar T, const scalar p, scalar& rho, scalar rhoU[3], scalar& rhoE);
    void getFlux(const scalar U[3], const scalar T, const scalar p, const uscalar N[3], scalar& rhoFlux, scalar rhoUFlux[3], scalar& rhoEFlux);

    void equation(const arr& rho, const arr& rhoU, const arr& rhoE, arr& drho, arr& drhoU, arr& drhoE);
    template<typename dtype>
    void boundary(const Boundary& boundary, arrType<dtype>& phi);

    void setMesh(Mesh const* mesh)  {
        this->mesh = mesh;
        this->interpolate = new Interpolator(mesh);
        this->operate = new Operator(mesh);
        this->boundaries = new Boundary[3];
    }

    ~RCF() {
        delete this->interpolate;
        delete this->operate;
        delete[] this->boundaries;
    }
};

#endif
