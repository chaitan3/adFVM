#ifndef COMMON_HPP
#define COMMON_HPP

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <limits>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <queue>
#include <tuple>
#include <map>
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


class MemoryBuffer {
    public:
    int usage;
    int maxUsage;
    // id to memory locs
    map<int, void *> shared;
    // reuse id to memory locs: uses pool
    map<string, void *> reuse;
    // size to queue of memory locs
    map<int, queue<void*>> pool;
    MemoryBuffer() {
    }
    
    virtual void alloc(void** data, int size) {
    }
    virtual void dealloc(void *data) {
    }

    void inc_mem(int size) {
        this->usage += size;
        if (this->usage > this->maxUsage) {
            this->maxUsage = this->usage;
        }
    }

    void dec_mem(int size) {
        this->usage -= size;
    }

    ~MemoryBuffer() {
        for (auto& kv : this->shared) { 
            this->dealloc(kv.second);
        }
        for (auto& kv : this->reuse) { 
            this->dealloc(kv.second);
        }
        for (auto& kv : this->pool) { 
            queue<void*>& q = kv.second;
            while(!q.empty()) {
                this->dealloc(q.front());
                q.pop();
            }
        }
    }
};

class CPUMemoryBuffer: public MemoryBuffer {
    public:
    void alloc(void **data, int size) {
        *data = malloc(size);
    }
    void dealloc(void* data) {
        free(data);
    }
    
};

template<typename MemoryBufferType>
class Memory {
    public:
    static MemoryBufferType mem;

};
template<typename MemoryBufferType>
MemoryBufferType Memory<MemoryBufferType>::mem = MemoryBufferType();

#ifdef _OPENMP
    #include <omp.h>
    //double __sync_fetch_and_add(scalar *operand, scalar incr) {
    //union {
    //    double   d;
    //    uint64_t i;
    //} oldval, newval, retval;
    //do {
    //    oldval.d = *(volatile double *)operand;
    //    newval.d = oldval.d + incr;
    //    __asm__ __volatile__ ("lock; cmpxchgq %1, (%2)"
    //      : "=a" (retval.i)
    //      : "r" (newval.i), "r" (operand),
    //       "0" (oldval.i)
    //      : "memory");
    //    } while (retval.i != oldval.i);
    //    return oldval.d;
    //}
#else
    #define omp_get_thread_num() 0
#endif

template<template<typename, integer, integer, integer> class derivedArrType, typename MemoryBufferType, typename dtype, integer shape1, integer shape2, integer shape3>
class baseArrType {
    public:
    dtype type;
    dtype* data;
    //integer dims;
    integer shape;
    integer size;
    integer bufSize;
    integer strides[NDIMS];
    bool ownData;
    bool keepMemory;
    int64_t id;

    derivedArrType<dtype, shape1, shape2, shape3>& self() { return static_cast<derivedArrType<dtype, shape1, shape2, shape3>&>(*this); }

    MemoryBufferType* get_mem() {
        return &(derivedArrType<dtype, shape1, shape2, shape3>::mem);
    }

    void set_default_values() {
        this->shape = 0;
        for (integer i = 0; i < NDIMS; i++)  {
            this->strides[i] = 0;
        }
        this -> size = 0;
        this -> bufSize = 0;
        this -> data = NULL;
        this -> ownData = false;
        this -> keepMemory = false;
        this -> id = 0;
    }

    void init(const integer shape) {
        this->set_default_values();
        this->shape = shape;
        //integer temp = 1;
        this->strides[3] = 1;
        this->strides[2] = shape3*this->strides[3];
        this->strides[1] = shape2*this->strides[2];
        this->strides[0] = shape1*this->strides[1];
        //cout << endl;
        this->size = this->strides[0]*shape;
        this->bufSize = this->size*sizeof(dtype);
    }

    void init_shape(const integer shape, bool zero, bool keepMemory, int64_t id) {
        this->init(shape);
        this->keepMemory = keepMemory;
        this->id = id;
        if (id != 0) {
            zero = (!this->shared_acquire()) && zero;
        } else {
            this->pool_acquire();
        }
        if (zero) {
            self().zero();
        }
    }
    baseArrType () {
        this->set_default_values();
    }

    baseArrType(const integer shape, bool zero=false, bool keepMemory=false, int64_t id=0) {
        this->init_shape(shape, zero, keepMemory, id);
    }

    baseArrType(const integer shape, dtype* data) {
        this -> init(shape);
        this -> data = data;
    }

    baseArrType(const integer shape, const dtype* data) {
        this->init(shape);
        this->data = const_cast<dtype *>(data);
    }

    baseArrType(const integer shape, const string& data) {
        this->init(shape);
        this->data = const_cast<dtype *>((dtype *)data.data());
    }

    // copy constructor?

    // move constructor
    baseArrType(baseArrType&& that) {
        //this->move(that);
        this->init(that.shape);
        this->data = that.data;
        this->ownData = that.ownData;
        this->keepMemory = that.keepMemory;
        this->id = that.id;
        that.ownData = false;
    }
    baseArrType& operator=(baseArrType&& that) {
        assert(this != &that);
        this->destroy();
        //this->move(that);
        this->init(that.shape);
        this->data = that.data;
        this->ownData = that.ownData;
        this->keepMemory = that.keepMemory;
        this->id = that.id;
        that.ownData = false;
        return *this;

    }

    void destroy() {
        if (this->ownData) {
            if (this->keepMemory) {
                this->pool_release();
            } else if (this->data != NULL) {
                this->get_mem()->dealloc((void*)this->data);

                this->data = NULL;
            }
        }
    }

    ~baseArrType() {
        this->destroy();
    };

    void pool_acquire() {
        int key = this->bufSize;
        if (this->get_mem()->pool.count(key) == 0) {
            this->get_mem()->pool[key] = queue<void*>();
        } 
        if (this->get_mem()->pool.at(key).empty()) {
            this->get_mem()->alloc((void**)&this->data, this->bufSize);  
        } else {
            this->data = (dtype *) this->get_mem()->pool[key].front();
            this->get_mem()->pool[key].pop();
        }
        //cout << "using " << this->data << endl;
        this->ownData = true;
    }
    void pool_release() {
        assert (this->ownData);
        int key = this->bufSize;
        this->get_mem()->pool[key].push((void *)this->data);
        this->ownData = false;
    }
    bool shared_acquire() {
        int64_t id = this->id;
        if (this->get_mem()->shared.count(id) > 0) {
            this->data = (dtype *)this->get_mem()->shared.at(id);
            return true;
        } else {
            this->get_mem()->alloc((void**)&this->data, this->bufSize);  
            this->get_mem()->shared[id] = (void *) this->data;
        }
        return false;
    }
    void reuse_acquire(string reuseId) {
        void* data = this->get_mem()->reuse[reuseId];
        this->data = (dtype*) data;
        int key = this->bufSize;
        this->ownData = true;
    }
    void reuse_release(string reuseId) {
        this->get_mem()->reuse[reuseId] = this->data;
        this->ownData = false;
    }
    

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
        return const_cast<dtype &>(static_cast<const derivedArrType<dtype, shape1, shape2, shape3> &>(*this)(i1));
    }
    dtype& operator()(const integer i1, const integer i2) {
        return const_cast<dtype &>(static_cast<const derivedArrType<dtype, shape1, shape2, shape3> &>(*this)(i1, i2));
    }
    dtype& operator()(const integer i1, const integer i2, const integer i3) {
        return const_cast<dtype &>(static_cast<const derivedArrType<dtype, shape1, shape2, shape3> &>(*this)(i1, i2, i3));
    }

    void zero() {
        throw logic_error("zero not implemented");
    }
    dtype* toHost() const {
        throw logic_error("toHost not implemented");
    }
    void toDevice(dtype *data) {
        throw logic_error("toDevice not implemented");
    }
};

template <typename dtype, integer shape1=1, integer shape2=1, integer shape3=1>
class arrType : public baseArrType<arrType, CPUMemoryBuffer, dtype, shape1, shape2, shape3>, public Memory<CPUMemoryBuffer> {
    public:
    using baseArrType<arrType, CPUMemoryBuffer, dtype, shape1, shape2, shape3>::baseArrType;

    void zero() {
        //cout << "cpu zero: " << endl;
        memset(this->data, 0, this->bufSize);
    }

    dtype* toHost() const {
        dtype* hdata = new dtype[this->size];
        memcpy(hdata, this->data, this->bufSize);
        return hdata;
    }
    void toDevice(dtype *data) {
        //memcpy(this->data, data, this->bufSize);
    }

    void copy(const integer index, const dtype* sdata, const integer n) {
        //cout << "cpu copy: " << endl;
        memcpy(&(*this)(index), sdata, n*sizeof(dtype));
    }

    void extract(const integer index, const integer* indices, const dtype* phiBuf, const integer n) {
        integer i, j, k;
        #pragma omp parallel for private(i, j, k)
        for (i = 0; i < n; i++) {
            integer p = indices[i];
            integer b = index + i;
            for (j = 0; j < shape1; j++) {
                for (k = 0; k < shape2; k++) {
                    (*this)(b, j, k) += phiBuf[p*shape1*shape2 + j*shape2 + k];
                }
            }
        }
    }
    void extract(const integer *indices, const dtype* phiBuf, const integer n) {
        integer i, j, k;
        #pragma omp parallel for private(i, j, k)
        for (i = 0; i < n; i++) {
            integer p = indices[i];
            for (j = 0; j < shape1; j++) {
                for (k = 0; k < shape2; k++) {
                    #pragma omp atomic 
                    (*this)(p, j, k) += phiBuf[i*shape1*shape2 + j*shape2 + k];
                    //__sync_fetch_and_add(&(*this)(p, j, k), phiBuf[i*shape1*shape2 + j*shape2 + k]);
                }
            }
        }
    }
    void info(int s=0, int e=-1) const {
        cout << setprecision(15) << this->sum(s, e) << endl;
        return;

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
    bool checkNAN() {
        for (integer i = 0; i < this->size; i++) {
            if (std::isnan(this->data[i])) {
                return true;
            }
        }
        return false;
    }
     
    dtype sum(int s=0,int  e=-1) const {
        dtype res = 0;
        if (e == -1) {
            e = this->shape;
        }
        for (integer i = s; i < e; i++) {
            for (integer j = 0; j < shape1; j++) {
                for (integer k = 0; k < shape2; k++) {
                    res += (*this)(i, j, k);
                }
            }
        }
        return res;
    }
};

//template<typename dtype, integer shaep1, integer shape2, integer shape3>


#ifndef GPU
    #define extArrType arrType
    typedef extArrType<scalar, 1> ext_vec;
#endif

typedef arrType<scalar> vec;
typedef arrType<scalar, 3> mat;
typedef arrType<integer> ivec;
typedef arrType<integer, 3> imat;

#endif
