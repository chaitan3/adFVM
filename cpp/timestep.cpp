#include "timestep.hpp"

/*void timeStepper(RCF &rcf, const arr& rho, const arr& rhoU, const arr& rhoE, arr& rhoN, arr& rhoUN, arr& rhoEN, scalar t, scalar dt) {*/
    //const Mesh& mesh = *rcf.mesh;

    //arr drho(rho.shape);
    //arr drhoU(rhoU.shape);
    //arr drhoE(rhoE.shape);
    //rcf.equation(rho, rhoU, rhoE, drho, drhoU, drhoE);

    //for (integer i = 0; i < mesh.nInternalCells; i++) {
        //rhoN(i) = rho(i) - dt*drho(i);
        //for (integer j = 0; j < 3; j++) {
            //rhoUN(i, j) = rhoU(i, j) - dt*drhoU(i, j);
        //}
        //rhoEN(i) = rhoE(i) - dt*drhoE(i);
    //}
/*}*/

void timeStepper(RCF *rcf, const arr& rho, const arr& rhoU, const arr& rhoE, arr& rhoN, arr& rhoUN, arr& rhoEN, scalar t, scalar dt) {
    const Mesh& mesh = *(rcf->mesh);

    const integer n = 3;
    scalar alpha[n][n] = {{1,0,0},{3./4, 1./4, 0}, {1./3, 0, 2./3}};
    scalar beta[n][n] = {{1,0,0}, {0,1./4,0},{0,0,2./3}};
    scalar gamma[n] = {0, 1, 0.5};

    arr rhos[n+1] = {{rho.shape, rho.data}, {rho.shape}, {rho.shape}, {rho.shape, rhoN.data}};
    arr rhoUs[n+1] = {{rhoU.shape, rhoU.data}, {rhoU.shape}, {rhoU.shape}, {rhoU.shape, rhoUN.data}};
    arr rhoEs[n+1] = {{rhoE.shape, rhoE.data}, {rhoE.shape}, {rhoE.shape}, {rhoE.shape, rhoEN.data}};
    arr drho(rho.shape);
    arr drhoU(rhoU.shape);
    arr drhoE(rhoE.shape);

    for (integer stage = 0; stage < n; stage++) {
        //solver.t = solver.t0 + gamma[i]*solver.dt
        rcf->equation(rhos[stage], rhoUs[stage], rhoEs[stage], drho, drhoU, drhoE);
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
}
