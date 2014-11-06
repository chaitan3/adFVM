#ifndef COMMON_HPP
#define COMMON_HPP

#define EIGEN_MATRIXBASE_PLUGIN "/home/talnikar/adFVM/cpp/ext.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <chrono>

using namespace Eigen;
using namespace std;

typedef ArrayXXd arr;
typedef Array<int32_t, Dynamic, Dynamic> iarr;
typedef SparseMatrix<double> spmat;

// switch to lambda funcs?
template <typename Derived, typename OtherDerived>
inline arr slice(const DenseBase<Derived>& array, const DenseBase<OtherDerived>& indices) {
    arr sliced(array.rows(), indices.cols());
    for (int i = 0; i < indices.cols(); i++) {
        sliced.col(i) = array.col(indices(0, i));
    }
    return sliced;
}

inline arr outerProduct(const arr& X, const arr& Y) {
    arr product(X.rows()*Y.rows(), X.cols());
    for (int i = 0; i < X.cols(); i++) {
        MatrixXd A = X.col(i).matrix() * Y.col(i).matrix().transpose();
        //MatrixXd A = Y.col(i).matrix() * X.col(i).matrix().transpose();
        VectorXd B(Map<VectorXd>(A.data(), A.cols()*A.rows()));
        product.col(i) = B;
    }
    return product;
}

inline arr transpose(const arr& X) {
    arr res = X;
    res.row(1) = X.row(3);
    res.row(2) = X.row(6);
    res.row(3) = X.row(1);
    res.row(5) = X.row(7);
    res.row(6) = X.row(2);
    res.row(7) = X.row(5);
    return res;
}

inline arr tdot(const arr& X, const arr& Y) {
    int rows = X.rows()/Y.rows();
    arr res = arr::Zero(rows, X.cols());
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < Y.rows(); j++) {
            res.row(i) += X.row(i+j*Y.rows())*Y.row(j);
        }
    }
    return res;
}

inline arr trace(const arr& phi) {
    arr res(1, phi.cols());
    res.row(0) = phi.row(0) + phi.row(4) + phi.row(8);
    return res;
}

// switch to lambda functions?
#define SELECT(X, i, n) ((X).block(0, (i), (X).rows(), (n)))
#define ROWMUL(X, Y) ((X).rowwise() * (Y).row(0))
#define DOT(X, Y) (((X) * (Y)).colwise().sum())
#define ROWDIV(X, Y) ((X).rowwise() / (Y).row(0))

#define SMALL 1e-30

#endif
