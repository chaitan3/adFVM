#ifndef COMMON_HPP
#define COMMON_HPP

#define EIGEN_MATRIXBASE_PLUGIN "/home/talnikar/adFVM/cpp/ext.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <chrono>
#include <valgrind/callgrind.h>

using namespace Eigen;
using namespace std;

typedef ArrayXXd arr;
typedef Array<int32_t, Dynamic, Dynamic> iarr;
typedef SparseMatrix<double> spmat;

// switch to lambda functions?
#define SELECT(X, i, n) ((X).block(0, (i), (X).rows(), (n)))
#define ROWMUL(X, Y) ((X).rowwise() * (Y).row(0))
#define ROWDIV(X, Y) ((X).rowwise() / (Y).row(0))
#define DOT(X, Y) (((X) * (Y)).colwise().sum())

// switch to lambda funcs?
template <typename Derived, typename OtherDerived>
inline arr slice(const DenseBase<Derived>& array, const DenseBase<OtherDerived>& indices) {
    arr sliced(array.rows(), indices.cols());
    for (int j = 0; j < indices.cols(); j++) {
        for (int i = 0; i < array.rows(); i++) {
            sliced(i, j) = array(i, indices(0, j));
        }
    }
    return sliced;
}

inline arr outerProduct(const arr& X, const arr& Y) {
    arr product(X.rows()*Y.rows(), X.cols());
    for (int j = 0; j < X.cols(); j++) {
        for (int i = 0; i < product.rows(); i++) {
            product(i, j)= X(i%3, j)*Y(i/3, j);
        }
    }
    return product;
}

inline arr transpose(const arr& X) {
    arr res = X;
    res.row(1).swap(res.row(3));
    res.row(2).swap(res.row(6));
    res.row(5).swap(res.row(7));
    return res;
}

inline arr tdot(const arr& X, const arr& Y) {
    int rows = X.rows()/Y.rows();
    arr res = arr::Zero(rows, X.cols());
    for (int k = 0; k < X.cols();  k++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < Y.rows(); j++) {
                res(i, k) += X(i+j*Y.rows(), k)*Y(j, k);
            }
        }
    }
    return res;
}

inline arr trace(const arr& phi) {
    arr res(1, phi.cols());
    res.row(0) = phi.row(0) + phi.row(4) + phi.row(8);
    return res;
}

#define SMALL 1e-30

#endif
