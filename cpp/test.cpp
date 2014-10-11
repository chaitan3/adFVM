#include "interp.hpp"

int main(int argc, char **argv) {
    Mesh mesh("../tests/cylinder");
    Interpolator interpolate(mesh);
    Field U("U", mesh, 2.0);
    arr phiF = interpolate.central(U.field);
}
