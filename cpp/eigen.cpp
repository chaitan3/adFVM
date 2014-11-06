#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

void dostuff()
{
    int Ni = 5, Nj = 3;
    ArrayXXf X = MatrixXf::Random(Ni, Nj);
    cout << X.rows() << endl;
    cout << X.cols() << endl;
    cout << X.colwise().sum().rows() << endl;
    cout << X.colwise().sum().cols() << endl;

    ArrayXXf Y(1, 3);
    Y << 1, 2, 3;

    cout << X << endl << endl << endl;
    cout << X.row(0) << endl << endl;
    cout << Y.row(0) << endl << endl;
    cout << X.rowwise()*Y.row(0) << endl << endl;
    cout << X.rowwise()/Y.row(0) << endl;

}
        

int main() {
    dostuff();
}
