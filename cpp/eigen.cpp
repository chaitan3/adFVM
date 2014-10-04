#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

inline VectorXd slice(const VectorXd &a, const VectorXi &b) {
    const int size = b.size();
    VectorXd c(size);
    for (int i = 0; i < size; i++) {
        c(i) = a(b(i));
    }
    return c;
}

void dostuff()
{
    int Ni = 50, Nj = 20;
    VectorXd x = VectorXd::LinSpaced(Ni+1, 20, -20);
    VectorXd y = VectorXd::LinSpaced(Nj+1, 5, -5);
    VectorXd a = VectorXd::Ones(Ni+1);
    a = (x.array().abs() < 10).select(1-(1+x.array()/10*3.1416)*0.1, a);
    MatrixXd X = x.replicate(Nj,1);
    MatrixXd Y = y.replicate(1,Ni);
}
        

int main() {
    dostuff();
}
