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

template <typename Derived, typename OtherDerived>
inline arr slice(const DenseBase<Derived>& array, const DenseBase<OtherDerived>& indices) {
    arr sliced(array.rows(), indices.cols());
    for (int i = 0; i < indices.cols(); i++) {
        sliced.col(i) = array.col(indices(0, i));
    }
    return sliced;
}

#endif
