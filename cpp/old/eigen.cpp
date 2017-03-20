#include <iostream>
#include <Eigen/Dense>
#include <chrono>

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

    int r = 1; int s = 3;
    int n = 100000;
    Array<double, Dynamic, Dynamic, RowMajor> A = ArrayXXd::Random(n, r);
    Array<double, Dynamic, Dynamic, RowMajor> B = ArrayXXd::Random(n, s);
    Array<double, Dynamic, Dynamic, RowMajor> C;

    auto start = chrono::system_clock::now();
    for (int i = 0; i < 100; i++) {
        C = B.colwise() * A.col(0);
    }
    auto end = chrono::system_clock::now();
    double time = ((double)chrono::duration_cast<chrono::milliseconds>(end - start).count())/1000;
    cout << time << endl;

}
        

int main() {
    dostuff();
}
