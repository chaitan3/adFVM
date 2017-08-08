#ifndef COMMON_HPP
#define COMMON_HPP

#include <iostream>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <tuple>
#include <limits>
#include <iomanip>
#include <vector>
#include <cassert>
#include <typeinfo>

using namespace std;

//#define ADIFF 1

#ifdef GPU
    typedef float scalar;
#else
    typedef double scalar;
#endif
typedef int32_t integer;

#define NDIMS 4

template <typename dtype, integer shape1=1, integer shape2=1, integer shape3=1>
class arrType {
    public:
    dtype* data;
    //integer dims;
    integer shape;
    integer size;
    integer strides[NDIMS];
    bool ownData;

    void init(const integer shape) {
        this->shape = shape;
        //integer temp = 1;
        this->strides[3] = 1;
        this->strides[2] = shape3*this->strides[3];
        this->strides[1] = shape2*this->strides[2];
        this->strides[0] = shape1*this->strides[1];
        //cout << endl;
        this -> size = this->strides[0]*shape;
        this -> ownData = true;
    }

    void destroy() {
        if (this->ownData && this->data != NULL) {
            delete[] this -> data; 
            this -> data = NULL;
        }
    }

    void move(arrType&& that) {
        this->init(that.shape);
        this->data = that.data;
        that.ownData = false;
    }

    arrType () {
        this->shape = 0;
        for (integer i = 0; i < NDIMS; i++)  {
            this->strides[i] = 0;
        }
        this -> size = 0;
        this -> data = NULL;
        this -> ownData = false;
    }

    arrType(const integer shape, bool zero=false) {
        this -> init(shape);
        this -> data = new dtype[this->size];
        if (zero) 
            this -> zero();
    }

    arrType(const integer shape, dtype* data) {
        this -> init(shape);
        this -> data = data;
        this -> ownData = false;
    }

    arrType(const integer shape, const dtype* data) {
        this->init(shape);
        this->data = const_cast<dtype *>(data);
        this->ownData = false ;
    }

    arrType(const integer shape, const string& data) {
        this->init(shape);
        this->data = const_cast<dtype *>((dtype *)data.data());
        this->ownData = false ;
    }

    // copy constructor?

    // move constructor
    arrType(arrType&& that) {
        //this->move(that);
        this->init(that.shape);
        this->data = that.data;
        this->ownData = that.ownData;
        that.ownData = false;
    }
    arrType& operator=(arrType&& that) {
        assert(this != &that);
        this->destroy();
        //this->move(that);
        this->init(that.shape);
        this->data = that.data;
        this->ownData = that.ownData;
        that.ownData = false;
        return *this;

    }
    ~arrType() {
        this->destroy();
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
    void info() const {
        scalar minPhi, maxPhi;
        minPhi = 1e100;
        maxPhi = -1e100;
        integer minLoc = -1, maxLoc = -1;
        for (integer i = 0; i < this->size; i++) {
            if (this->data[i] < minPhi) {
                minPhi = this->data[i];
                minLoc = i;
            }
            if (this->data[i] > maxPhi) {
                maxPhi = this->data[i];
                maxLoc = i;
            }

        }
        cout << "phi min/max:" << minPhi << " " << maxPhi << endl;
        //cout << "loc min/max:" << minLoc << " " << maxLoc << endl;
    } 

    void zero() {
        memset(this->data, 0, this->size*sizeof(dtype));
    }
    dtype sum() {
        dtype s = 0;
        for (integer i = 0; i < this->size; i++) {
            s += this->data[i];
        }
        return s;
    }

};

typedef arrType<scalar> vec;
typedef arrType<scalar, 3> mat;

typedef arrType<integer> ivec;
typedef arrType<integer, 3> imat;

#define SMALL 1e-30

#endif
