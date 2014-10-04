#include "field.hpp"

int main(int argc, char **argv) {
    Mesh mesh("../tests/cylinder");
    Field field("U", mesh, 2.0);
}
