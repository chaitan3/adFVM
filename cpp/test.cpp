#include "field.hpp"
#include "interp.hpp"
#include "op.hpp"

int main(int argc, char **argv) {
    Mesh mesh("../tests/cyclic");
    Interpolator interpolate(mesh);
    Operator operate(mesh);
    Field U("U", mesh, 0.1);
    Field T("T", mesh, 0.1);
    double dt = 1e-5;
    for (int i = 0; i < 100; i++) {
        cout << i << endl;
        arr UF = interpolate.central(U.field);
        arr TF = interpolate.central(T.field);
        arr TUFdotN = DOT(ROWMUL(UF, TF), mesh.normals);
        T.field.leftCols(mesh.nInternalCells) -= (operate.div(TUFdotN) - operate.laplacian(T.field)) * dt;
        T.updateGhostCells();
    }
    T.write(1.0);
    return 0;
}
