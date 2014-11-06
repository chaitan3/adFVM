#include "field.hpp"
#include "interp.hpp"
#include "op.hpp"

class RCF {
    
    double R = 8.314;
    //double Cp = 1004.5;
    double Cp = 2.5;
    double gamma = 1.4;
    double Cv = Cp/gamma;
    double Pr = 0.7;
    double CFL = 0.2;
    double stepFactor = 1.2;

    Mesh mesh;

    // confirm that make_tuple doesn't create copies
    tuple<arr, arr, arr> primitive(const arr& rho, const arr& rhoU, const arr& rhoE) {
        arr U = ROWDIV(rhoU, rho);
        arr e = rhoE/rho - 0.5*DOT(U, U);
        return make_tuple(U, e/this->Cv, (this->gamma-1)*rho*e);
    }
    
    tuple<arr, arr, arr> conservative(const arr& U, const arr& T, const arr& p) {
        arr e = this->Cv * T;
        arr rho = p/(e*(this->gamma - 1));
        return make_tuple(rho, ROWMUL(U, rho), rho*(e + 0.5*DOT(U, U)));
    }

    inline Ref<arr> internalField(arr& phi) {
        Ref<arr> phiI = SELECT(phi, 0, mesh.nInternalCells);
        return phiI;
    }

    inline Ref<arr> boundaryField(arr& phi) { 
        Ref<arr> phiB = SELECT(phi, mesh.nInternalCells, mesh.nGhostCells);
        return phiB;
    }
    
    inline arr mu(const arr& T) {
        return 0*T;//1.4792e-06*T.pow(1.5)/(T+116);
    }
    inline arr kappa(const arr& mu, const arr& T) {
        return mu*this->Cp/this->Pr;
    }
    
    public:
        RCF (string caseDir, double time):
            mesh(caseDir) {

            Interpolator interpolate(mesh);

            Operator operate(mesh);

            Field U("U", mesh, 0.0);
            Field T("T", mesh, 0.0);
            Field p("p", mesh, 0.0);
            double dt = 1.0;
            double t = 0.0;

            arr pos = arr::Ones(1, mesh.nFaces);
            arr neg = -pos;
            arr rho, rhoU, rhoE;
            tie(rho, rhoU, rhoE) = this->conservative(U.field, T.field, p.field);

            printf("\n");
            for (int i = 0; i < 10001; i++) {
                printf("Iteration count: %d\n", i);
                auto start = chrono::system_clock::now();

                printf("rho: min: %f max: %f\n", rho.minCoeff(), rho.maxCoeff());
                printf("rhoU: min: %f max: %f\n", rhoU.minCoeff(), rhoU.maxCoeff());
                printf("rhoE: min: %f max: %f\n", rhoE.minCoeff(), rhoE.maxCoeff());
                arr gradRho = Field(mesh, operate.grad(interpolate.central(rho))).field;
                arr gradRhoU = Field(mesh, operate.grad(interpolate.central(rhoU))).field;
                arr gradRhoE = Field(mesh, operate.grad(interpolate.central(rhoE))).field;
                arr rhoLF = interpolate.TVD(rho, gradRho, pos);
                arr rhoRF = interpolate.TVD(rho, gradRho, neg);
                arr rhoULF = interpolate.TVD(rhoU, gradRhoU, pos);
                arr rhoURF = interpolate.TVD(rhoU, gradRhoU, neg);
                arr rhoELF = interpolate.TVD(rhoE, gradRhoE, pos);
                arr rhoERF = interpolate.TVD(rhoE, gradRhoE, neg);
                arr ULF, URF, TLF, TRF, pLF, pRF;
                tie(ULF, TLF, pLF) = this->primitive(rhoLF, rhoULF, rhoELF);
                tie(URF, TRF, pRF) = this->primitive(rhoRF, rhoURF, rhoERF);

                //cRF not used?
                arr cLF = sqrt(this->gamma*pLF/rhoLF);
                arr cRF = sqrt(this->gamma*pRF/rhoRF);
                arr UnLF = DOT(ULF, mesh.normals);
                arr UnRF = DOT(URF, mesh.normals);
                arr cF(4, mesh.nFaces);
                cF.row(0) = (UnLF + cLF).row(0);
                cF.row(1) = (UnRF + cLF).row(0);
                cF.row(2) = (UnLF - cLF).row(0);
                cF.row(3) = (UnRF - cLF).row(0);
                arr aF = cF.abs().colwise().maxCoeff();

                arr rhoFlux = 0.5*(rhoLF*UnLF + rhoRF*UnRF) - 0.5*aF*(rhoRF-rhoLF);
                //self.flux = 2*rhoFlux/(rhoLF + rhoRF)
                arr rhoUFlux = 0.5*(ROWMUL(rhoULF, UnLF) + ROWMUL(rhoURF, UnRF)) - 0.5*ROWMUL((rhoURF-rhoULF), aF);
                arr rhoEFlux = 0.5*((rhoELF + pLF)*UnLF + (rhoERF + pRF)*UnRF) - 0.5*aF*(rhoERF-rhoELF);
                arr pF = 0.5*(pLF + pRF);
                
                //optim?
                arr TF = 0.5*(TLF + TRF);
                arr mu = this->mu(TF);
                arr kappa = this->kappa(mu, TF);
                arr UF = 0.5*(ULF + URF);
                arr gradUF = interpolate.central(Field(mesh, operate.grad(UF)).field);
                arr sigmaF = ROWMUL((operate.snGrad(U.field) + tdot(gradUF, mesh.normals) - (2./3)*ROWMUL(mesh.normals, trace(gradUF))), mu);

                arr drho = operate.div(rhoFlux);
                arr drhoU = operate.div(rhoUFlux) + operate.grad(pF) - operate.div(sigmaF);
                arr drhoE = operate.div(rhoEFlux) - (operate.laplacian(T.field, kappa) + operate.div(DOT(sigmaF, UF)));

                cLF = (UnLF + aF).abs()*0.5;
                cRF = (UnRF - aF).abs()*0.5;
                aF = (cLF > cRF).select(cLF, cRF);
                dt = min(dt*stepFactor, this->CFL*(mesh.deltas/aF).minCoeff());
                
                // integration
                Ref<arr> rhoI = this->internalField(rho);
                Ref<arr> rhoUI = this->internalField(rhoU);
                Ref<arr> rhoEI = this->internalField(rhoE);
                rhoI -= dt*drho;
                rhoUI -= dt*drhoU;
                rhoEI -= dt*drhoE;
                t += dt;
                printf("Simulation Time: %f Time step: %f\n", t, dt);

                // boundary correction
                Ref<arr> UI = this->internalField(U.field);
                Ref<arr> TI = this->internalField(T.field);
                Ref<arr> pI = this->internalField(p.field);
                tie(UI, TI, pI) = this->primitive(rhoI, rhoUI, rhoEI);
                U.updateGhostCells();
                T.updateGhostCells();
                p.updateGhostCells();

                Ref<arr> rhoB = this->boundaryField(rho);
                Ref<arr> rhoUB = this->boundaryField(rhoU);
                Ref<arr> rhoEB = this->boundaryField(rhoE);
                Ref<arr> UB = this->boundaryField(U.field);
                Ref<arr> TB = this->boundaryField(T.field);
                Ref<arr> pB = this->boundaryField(p.field);
                tie(rhoB, rhoUB, rhoEB) = this->conservative(UB, TB, pB);

                auto end = chrono::system_clock::now();
                double time = ((double)chrono::duration_cast<chrono::milliseconds>(end - start).count())/1000;
                printf("Time for iteration: %f\n\n", time);

                if (i % 500 == 0) {
                    U.write(t);
                    T.write(t);
                    p.write(t);
                }
            }
        }
};

int main(int argc, char **argv) {
    // template arr to be scalar/vector? 
    // or maybe use tensor functionality in eigen

    RCF("../tests/forwardStep", 0.0);
    return 0;
}
