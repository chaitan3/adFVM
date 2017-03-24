#ifndef COMMON_HPP
#define COMMON_HPP

#include <iostream>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <tuple>

using namespace std;

//#define ADIFF 1

#ifdef ADIFF
    #include "codi.hpp"
    typedef codi::RealReverse scalar;
#else 
    #include "mpi.h"
    #define AMPI_Irecv MPI_Irecv
    #define AMPI_Isend MPI_Isend
    #define AMPI_Waitall MPI_Waitall
    #define AMPI_Request MPI_Request
    typedef double scalar;
#endif

typedef double uscalar;
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
    bool ownData;

    void init(const integer* shape) {
        copy(shape, shape + NDIMS, this->shape);
        integer temp = 1;
        for (integer i = NDIMS-1; i >= 0; i--) {
            //cout << shape[i] << " ";
            this->strides[i] = temp;
            temp *= shape[i];
        }
        //cout << endl;
        this -> size = temp;
        this -> ownData = true;
    }

    arrType () {
        for (integer i = 0; i < NDIMS; i++)  {
            this->shape[i] = 0;
            this->strides[i] = 0;
        }
        this -> size = 0;
        this -> data = NULL;
        this -> ownData = false;
    }

    arrType(const integer* shape) {
        this -> init(shape);
        this -> data = new dtype[this->size];
    }

    arrType(const integer* shape, dtype* data) {
        this -> init(shape);
        this -> data = data;
        this -> ownData = false;
    }

    arrType(const integer shape1) {
        const integer shape[NDIMS] = {shape1, 1, 1, 1};
        this -> init(shape);
        this -> data = new dtype[this->size];
    }
    arrType(const integer shape1, const integer shape2) {
        const integer shape[NDIMS] = {shape1, shape2, 1, 1};
        this -> init(shape);
        this -> data = new dtype[this->size];
    }
    arrType(const integer shape1, const integer shape2, const integer shape3) {
        const integer shape[NDIMS] = {shape1, shape2, shape3, 1};
        this -> init(shape);
        this -> data = new dtype[this->size];
    }

    arrType(const integer* shape, const dtype* data) {
        this->init(shape);
        this->data = const_cast<dtype *>(data);
        this->ownData = false ;
    }

    arrType(const integer* shape, const string& data) {
        this->init(shape);
        this->data = const_cast<dtype *>((dtype *)data.data());
        this->ownData = false ;
    }


    //arrType(const arrType& ref) {
    //    cout << "copy arr " << ref.shape[0] << " " << ref.shape[1] << endl;
    //}
    //arrType& operator= (arrType& ref) {
    //    cout << "assign arr " << ref.shape[0] << " " << ref.shape[1] << endl;
    //    ref.ownData = false;
    //    this -> init(ref.shape);
    //    this -> data = data;
    //    this -> ownData = false;
    //}

    ~arrType() {
        //if (this->ownData) {
        if (this->ownData && this->data != NULL) {
            delete[] this -> data; 
            this -> data = NULL;
        }
    };
    
    const dtype& operator() (const integer i1) const {
        return const_cast<const dtype &>(data[i1*this->strides[0]]);
    }

    const dtype& operator() (const integer i1, const integer i2) const {
        return const_cast<const dtype &>(data[i1*this->strides[0] + 
                      i2*this->strides[1]]);
    }
    const dtype& operator() (const integer i1, const integer i2, const integer i3) const {
        return const_cast<const dtype &>(data[i1*this->strides[0] + 
                      i2*this->strides[1] +
                      i3*this->strides[2]]);
    }



    dtype& operator()(const integer i1) {
        return const_cast<dtype &>(static_cast<const arrType &>(*this)(i1));
    }
    dtype& operator()(const integer i1, const integer i2) {
        return const_cast<dtype &>(static_cast<const arrType &>(*this)(i1, i2));
    }
    dtype& operator()(const integer i1, const integer i2, const integer i3) {
        return const_cast<dtype &>(static_cast<const arrType &>(*this)(i1, i2, i3));
    }

    bool checkNAN() {
        for (integer i = 0; i < this->size; i++) {
            if (std::isnan(this->data[i])) {
                return true;
            }
        }
        return false;
    }
    void info() {
        scalar minPhi, maxPhi;
        minPhi = 1e100;
        maxPhi = -1e100;
        for (integer i = 0; i < this->size; i++) {
            if (this->data[i] < minPhi) {
                minPhi = this->data[i];
            }
            if (this->data[i] > maxPhi) {
                maxPhi = this->data[i];
            }

        }
        cout << "phi " << minPhi << " " << maxPhi << endl;
    }

    void zero() {
        for (integer i = 0; i < this->size; i++) {
            this->data[i] = 0;
        }
    }

    #ifdef ADIFF 
        void adInit(codi::RealReverse::TapeType& tape) {
            uscalar *udata = (uscalar *)this->data;
            this -> data = new scalar[this->size];
            for (integer i = 0; i < this->size; i++) {
                this -> data[i] = udata[i];
                tape.registerInput(this->data[i]);
            }
            this->ownData = true;
        }
        void adGetGrad(const arrType<scalar>& phi) {
            for (integer i = 0; i < this->size; i++) {
                this -> data[i] = phi.data[i].getGradient();
            }
            this->ownData = true;
        }
    #endif
};

typedef arrType<scalar> arr;
typedef arrType<uscalar> uarr;
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
