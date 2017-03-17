#ifndef COMMON_HPP
#define COMMON_HPP

#include <iostream>
#include <cstdint>
#include <tuple>

using namespace std;

typedef double scalar;
typedef int32_t integer;

#define NDIMS 4

template <typename dtype>
class arrType {
    public:
    dtype* data;
    //integer dims;
    integer shape[NDIMS];
    integer size;
    integer strides[NDIMS];

    void init(const integer* shape, dtype* data) {
        copy(shape, shape + NDIMS, this->shape);
        integer temp = 1;
        for (integer i = NDIMS-1; i >= 0; i--) {
            this->strides[i] = temp;
            temp *= shape[i];
        }
        this -> size = temp;
        this -> data = data;
    }

    arrType () {}

    arrType(const integer* shape) {
        data = new dtype[size];
        this -> init(shape, data);
    }

    arrType(const integer* shape, dtype* data) {
        this -> init(shape, data);
    }

    ~arrType() {
        delete this -> data; 
    };
    
    const dtype& operator() (const integer i1) const {
        return const_cast<const dtype &>(*(data + i1*sizeof(dtype)*this->strides[0]));
    }

    const dtype& operator() (const integer i1, const integer i2) const {
        return const_cast<const dtype &>(*(data + i1*sizeof(dtype)*this->strides[0] + 
                      i2*sizeof(dtype)*this->strides[1]));
    }

    dtype& operator()(const integer i1) {
        return const_cast<dtype &>(static_cast<const arrType &>(*this)(i1));
    }
};

typedef arrType<scalar> arr;
typedef arrType<integer> iarr;

// switch to lambda funcs?
//template <typename Derived, typename OtherDerived>
//inline arr slice(const DenseBase<Derived>& array, const DenseBase<OtherDerived>& indices) {
//    arr sliced(array.rows(), indices.cols());
//    for (int j = 0; j < indices.cols(); j++) {
//        for (int i = 0; i < array.rows(); i++) {
//            sliced(i, j) = array(i, indices(0, j));
//        }
//    }
//    return sliced;
//}
//
//inline arr outerProduct(const arr& X, const arr& Y) {
//    arr product(X.rows()*Y.rows(), X.cols());
//    for (int j = 0; j < X.cols(); j++) {
//        for (int i = 0; i < product.rows(); i++) {
//            product(i, j)= X(i%3, j)*Y(i/3, j);
//        }
//    }
//    return product;
//}
//
//inline arr transpose(const arr& X) {
//    arr res = X;
//    res.row(1).swap(res.row(3));
//    res.row(2).swap(res.row(6));
//    res.row(5).swap(res.row(7));
//    return res;
//}
//
//inline arr tdot(const arr& X, const arr& Y) {
//    int rows = X.rows()/Y.rows();
//    arr res = arr::Zero(rows, X.cols());
//    for (int k = 0; k < X.cols();  k++) {
//        for (int i = 0; i < rows; i++) {
//            for (int j = 0; j < Y.rows(); j++) {
//                res(i, k) += X(i+j*Y.rows(), k)*Y(j, k);
//            }
//        }
//    }
//    return res;
//}
//
//inline arr trace(const arr& phi) {
//    arr res(1, phi.cols());
//    res.row(0) = phi.row(0) + phi.row(4) + phi.row(8);
//    return res;
//}

#define SMALL 1e-30

#endif
