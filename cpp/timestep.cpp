#include "timestep.hpp"

tuple<scalar, scalar> euler(RCF *rcf, const vec& rho, const mat& rhoU, const vec& rhoE, vec& rhoN, mat& rhoUN, vec& rhoEN, scalar t, scalar dt) {
    const Mesh& mesh = *(rcf->mesh);

    vec drho(rho.shape);
    mat drhoU(rhoU.shape);
    vec drhoE(rhoE.shape);
    scalar objective, dtc;
    rcf->equation(rho, rhoU, rhoE, drho, drhoU, drhoE, objective, dtc);

    for (integer i = 0; i < mesh.nInternalCells; i++) {
        rhoN(i) = rho(i) - dt*drho(i);
        for (integer j = 0; j < 3; j++) {
            rhoUN(i, j) = rhoU(i, j) - dt*drhoU(i, j);
        }
        rhoEN(i) = rhoE(i) - dt*drhoE(i);
    }
    return make_tuple(objective, dtc);
}

tuple<scalar, scalar> SSPRK(RCF *rcf, const vec& rho, const mat& rhoU, const vec& rhoE, vec& rhoN, mat& rhoUN, vec& rhoEN, scalar t, scalar dt) {
    const Mesh& mesh = *(rcf->mesh);

    const integer n = 3;
    scalar alpha[n][n] = {{1,0,0},{3./4, 1./4, 0}, {1./3, 0, 2./3}};
    scalar beta[n][n] = {{1,0,0}, {0,1./4,0},{0,0,2./3}};
    scalar gamma[n] = {0, 1, 0.5};
    scalar objective[n], dtc[n];

    vec rhos[n+1] = {{rho.shape, rho.data}, {rho.shape}, {rho.shape}, {rho.shape, rhoN.data}};
    mat rhoUs[n+1] = {{rhoU.shape, rhoU.data}, {rhoU.shape}, {rhoU.shape}, {rhoU.shape, rhoUN.data}};
    vec rhoEs[n+1] = {{rhoE.shape, rhoE.data}, {rhoE.shape}, {rhoE.shape}, {rhoE.shape, rhoEN.data}};
    vec drho(rho.shape);
    mat drhoU(rhoU.shape);
    vec drhoE(rhoE.shape);

    for (integer stage = 0; stage < n; stage++) {
        //solver.t = solver.t0 + gamma[i]*solver.dt
        rcf->stage = stage;
        rcf->equation(rhos[stage], rhoUs[stage], rhoEs[stage], drho, drhoU, drhoE, objective[stage], dtc[stage]);
        integer curr = stage + 1;
        scalar b = beta[stage][stage];
        for (integer i = 0; i < mesh.nInternalCells; i++) {
            rhos[curr](i) = -b*drho(i)*dt;
            for (integer j = 0; j < 3; j++) {
                rhoUs[curr](i, j) = -b*drhoU(i, j)*dt;
            }
            rhoEs[curr](i) = -b*drhoE(i)*dt;
        }
        for (integer prev = 0; prev < curr; prev++) {
            scalar a = alpha[stage][prev];
            for (integer i = 0; i < mesh.nInternalCells; i++) {
                rhos[curr](i) += a*rhos[prev](i);
                for (integer j = 0; j < 3; j++) {
                    rhoUs[curr](i, j) += a*rhoUs[prev](i, j);
                }
                rhoEs[curr](i) += a*rhoEs[prev](i);
            }
        }
    }
    return make_tuple(objective[0], dtc[0]);
}
