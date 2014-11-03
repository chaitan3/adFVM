#ifndef COMMON_HPP
#define COMMON_HPP

#define EIGEN_MATRIXBASE_PLUGIN "/home/talnikar/adFVM/cpp/ext.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

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
        VectorXd B(Map<VectorXd>(A.data(), A.cols()*A.rows()));
        product.col(i) = B;
    }
    return product;
}

// switch to lambda functions?
#define SELECT(X, i, n) ((X).block(0, (i), (X).rows(), (n)))
#define ROWMUL(X, Y) ((X).rowwise() * (Y).row(0))
#define ROWDIV(X, Y) ((X).rowwise() / (Y).row(0))
#define DOT(X, Y) (((X) * (Y)).colwise().sum())

#define SMALL 1e-30

#endif
