#include "field.hpp"
#include "interp.hpp"
#include "op.hpp"

class RCF {
    
    double R = 8.314;
    double Cp = 1004.5;
    double gamma = 1.4;
    double Cv = Cp/gamma;
    double Pr = 0.7;
    double CFL = 0.6;
    double stepFactor = 1.2;

    tuple<arr, arr, arr> primitive(const arr& rho, const arr& rhoU, const arr& rhoE) {
        arr U = rhoU/rho;
        arr e = rhoE/rho - 0.5*DOT(U, U);
        return make_tuple(U, e/this->Cv, (this->gamma-1)*rho*e);
    }
    
    tuple<arr, arr, arr> conservative(const arr& U, const arr& T, const arr& p) {
        arr e = this->Cv * T;
        arr rho = p/(e*(this->gamma - 1));
        return make_tuple(rho, rho*U, rho*(e + 0.5*DOT(U, U)));
    }
    
    public:
        RCF (string caseDir, double time) {
            Mesh mesh(caseDir);

            Interpolator interpolate(mesh);
            Operator operate(mesh);

            Field U("U", mesh, 0.0);
            Field T("T", mesh, 0.0);
            Field p("p", mesh, 0.0);
            double dt = 1.0;

            arr pos = arr::Ones(1, mesh.nFaces);
            arr neg = -pos;
            arr rho, rhoU, rhoE;

            tie(rho, rhoU, rhoE) = this->conservative(U.field, T.field, p.field);
            arr gradRho = operate.grad(rho);
            arr gradRhoU = operate.grad(rhoU);
            arr gradRhoE = operate.grad(rhoE);
            // grad make it full
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
            arr rhoUFlux = 0.5*(rhoULF*UnLF + rhoURF*UnRF) - 0.5*aF*(rhoURF-rhoULF);
            arr rhoEFlux = 0.5*((rhoELF + pLF)*UnLF + (rhoERF + pRF)*UnRF) - 0.5*aF*(rhoERF-rhoELF);
            arr pF = 0.5*(pLF + pRF);
            
            //optim?
            arr TF = 0.5*(TLF + TRF);
            arr mu = this->mu(TF);
            arr kappa = this->kappa(mu, TF);
            arr UF = 0.5*(ULF + URF);
            arr gradUF = interpolate.central(operate.grad(UF));
            // fix dot, trace
            arr sigmaF = mu*(operate.snGrad(U.field) + DOT(gradUF, mesh.normals) - (2./3)*TRACE(gradUF)*mesh.normals);

            arr drho = operate.div(rhoFlux);
            arr drhoU = operate.div(rhoUFlux) + operate.grad(pF) - operate.div(sigmaF);
            arr drhoE = operate.div(rhoEFlux) - operate.laplacian(T.field, kappa) + operate.div(DOT(sigmaF, UF)) ;

            cLF = (UnLF + aF).abs()*0.5;
            cRF = (UnRF - aF).abs()*0.5;
            aF = (cLF > cRF).select(cLF, cRF);
            dt = min(dt*stepFactor, this->CFL*(mesh.deltas/aF).maxCoeff());
        }
};

int main(int argc, char **argv) {
    // template arr to be scalar/vector? 
    // or maybe use tensor functionality in eigen

    RCF("../tests/forwardStep", 0.0);
    return 0;
}
