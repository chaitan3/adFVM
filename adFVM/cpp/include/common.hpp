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
#include <map>
#include <cassert>
#include <typeinfo>

using namespace std;

//#define ADIFF 1#
//
#define STRING(s) #s
#define VALUE(x) STRING(x)
#define PRINT(var) #var "="  VALUE(var)

#if defined(GPU) || defined(CPU_FLOAT32)
    typedef float scalar;
#else
    typedef double scalar;
#endif
typedef int32_t integer;

#define NDIMS 4

struct memory {
    int usage;
    int maxUsage;
    map<string, void *> refs;
};
extern struct memory mem;

template <typename dtype, integer shape1=1, integer shape2=1, integer shape3=1>
class arrType {
    public:
    dtype type;
    dtype* data;
    //integer dims;
    integer shape;
    integer size;
    integer strides[NDIMS];
    bool ownData;
    bool staticVariable;

    void initPre() {
        this->shape = 0;
        for (integer i = 0; i < NDIMS; i++)  {
            this->strides[i] = 0;
        }
        this -> size = 0;
        this -> data = NULL;
        this -> ownData = false;
        this -> staticVariable = false;
    }

    void init(const integer shape) {
        this->initPre();
        this->shape = shape;
        //integer temp = 1;
        this->strides[3] = 1;
        this->strides[2] = shape3*this->strides[3];
        this->strides[1] = shape2*this->strides[2];
        this->strides[0] = shape1*this->strides[1];
        //cout << endl;
        this -> size = this->strides[0]*shape;
    }
    void destroy() {
        if (this->ownData && this->data != NULL) {
            this->dec_mem();
            delete[] this -> data; 
            this -> data = NULL;
        }
    }
    void inc_mem() {
        mem.usage += this->size*sizeof(dtype);
        if (mem.usage > mem.maxUsage) {
            mem.maxUsage = mem.usage;
        }
    }

    void dec_mem() {
        mem.usage -= this->size*sizeof(dtype);
    }

    arrType () {
        this->initPre();
    }

    arrType(const integer shape, bool zero=false) {
        this -> init(shape);
        this -> data = new dtype[this->size];
        this -> ownData = true;
        if (zero) 
            this -> zero();
        this->inc_mem();
    }

    arrType(const integer shape, dtype* data) {
        this -> init(shape);
        this -> data = data;
    }

    arrType(const integer shape, const dtype* data) {
        this->init(shape);
        this->data = const_cast<dtype *>(data);
    }

    //arrType(const integer shape, const string& data) {
    //    this->init(shape);
    //    this->data = const_cast<dtype *>((dtype *)data.data());
    //    this->ownData = false ;
    //}

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

    void copy(const integer index, const dtype* sdata, const integer n) {
        memcpy(&this->data[index], sdata, n*sizeof(dtype));
    }
    void extract(const integer index, const dtype* phiBuf, const integer* indices, const integer n) {
        for (integer i = 0; i < n; i++) {
            integer p = indices[i];
            integer b = index + i;
            for (integer j = 0; j < shape1; j++) {
                for (integer k = 0; k < shape2; k++) {
                    (*this)(b, j, k) += phiBuf[p*shape1*shape2 + j*shape2 + k];
                }
            }
        }
    }
    void extract(const integer *indices, const dtype* phiBuf, const integer n) {
        for (integer i = 0; i < n; i++) {
            integer p = indices[i];
            for (integer j = 0; j < shape1; j++) {
                for (integer k = 0; k < shape2; k++) {
                    (*this)(p, j, k) += phiBuf[i*shape1*shape2 + j*shape2 + k];
                }
            }
        }
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



#ifdef GPU
#define GPU_THREADS_PER_BLOCK 256
//#define GPU_THREADS_PER_BLOCK 512
//#define GPU_THREADS_PER_BLOCK 1024
#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template<typename dtype, integer shape1, integer shape2>
__global__ void _extract1(const integer n, dtype* phi1, const dtype* phi2, const integer* indices) {
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    if (i < n) {
        integer p = indices[i];
        for (integer j = 0; j < shape1; j++) {
            for (integer k = 0; k < shape2; k++) {
                 atomicAdd(&phi1[p*shape1*shape2 + j*shape2 + k], phi2[i*shape1*shape2 + j*shape2 + k]);
            }
        }
    }
}

template<typename dtype, integer shape1, integer shape2>
__global__ void _extract2(const integer n, dtype* phi1, const dtype* phi2, const integer* indices) {
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    if (i < n) {
        integer p = indices[i];
        for (integer j = 0; j < shape1; j++) {
            for (integer k = 0; k < shape2; k++) {
                 phi1[i*shape1*shape2 + j*shape2 + k] += phi2[p*shape1*shape2 + j*shape2 + k];
            }
        }
    }
}


template <typename dtype, integer shape1=1, integer shape2=1, integer shape3=1>
class gpuArrType: public arrType<dtype, shape1, shape2, shape3> {
    public:
    gpuArrType(const integer shape, bool zero=false) {
        this -> init(shape);
        gpuErrorCheck(cudaMalloc(&this->data, this->size*sizeof(dtype)));
        this->inc_mem();
        this->ownData = true;
        if (zero) 
            this -> zero();
    }
    gpuArrType(const integer shape, dtype* data) {
        this -> init(shape);
        this -> data = data;
    }
    void destroy() {
        if (this->ownData && this->data != NULL) {
            this->dec_mem();
            cudaFree(this->data);
            this -> data = NULL;
        }
    }
    void zero() {
        gpuErrorCheck(cudaMemset(this->data, 0, this->size*sizeof(dtype)));
    }
    void copy(integer index, dtype* sdata, integer n) {
        cudaMemcpy(&this->data[index], sdata, n*sizeof(dtype), cudaMemcpyDeviceToDevice);
    }
    void extract(const integer index, const dtype* phiBuf, const integer* indices, const integer n) {
        integer blocks = n/GPU_THREADS_PER_BLOCK + 1;
        integer threads = min(GPU_THREADS_PER_BLOCK, n);
        _extract1<dtype, shape1, shape2><<<blocks, threads>>>(n, &(*this)(index), phiBuf, indices);
    }
    void extract(const integer *indices, const dtype* phiBuf, const integer n) {
        integer blocks = n/GPU_THREADS_PER_BLOCK + 1;
        integer threads = min(GPU_THREADS_PER_BLOCK, n);
        _extract2<dtype, shape1, shape2><<<blocks, threads>>>(n, this->data, phiBuf, indices);
    }
    
    dtype* toHost() const {
        dtype* hdata = new dtype[this->size];
        gpuErrorCheck(cudaMemcpy(hdata, this->data, this->size*sizeof(dtype), cudaMemcpyDeviceToHost));
        return hdata;
    }
    void toDevice(dtype* data) {
        assert (this->data == NULL);
        this->inc_mem();
        gpuErrorCheck(cudaMalloc(&this->data, this->size*sizeof(dtype)));
        gpuErrorCheck(cudaMemcpy(this->data, data, this->size*sizeof(dtype), cudaMemcpyHostToDevice));
        this->ownData = true;
    }
    void info() const {
        dtype minPhi, maxPhi;
        minPhi = 1e30;
        maxPhi = -1e30;
        integer minLoc = -1, maxLoc = -1;
        dtype* hdata = this->toHost();
        for (integer i = 0; i < this->size; i++) {
            if (hdata[i] < minPhi) {
                minPhi = hdata[i];
                minLoc = i;
            }
            if (hdata[i] > maxPhi) {
                maxPhi = hdata[i];
                maxLoc = i;
            }

        }
        delete[] hdata;
        cout << "phi min/max:" << minPhi << " " << maxPhi << endl;
        //cout << "loc min/max:" << minLoc << " " << maxLoc << endl;
    }
    gpuArrType() : arrType<dtype, shape1, shape2, shape3>() {}
    gpuArrType& operator=(gpuArrType&& that) {
        assert(this != &that);
        this->destroy();
        //this->move(that);
        this->init(that.shape);
        this->data = that.data;
        this->ownData = that.ownData;
        that.ownData = false;
        return *this;

    }
    ~gpuArrType() {
        this->destroy();
    };
};

#define extArrType gpuArrType

#else

#define extArrType arrType

#endif

typedef extArrType<scalar, 1> ext_vec;

#endif
