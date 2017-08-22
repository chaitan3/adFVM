#ifndef GPU_HPP
#define GPU_HPP
#ifdef GPU
#include "common.hpp"

#define GPU_THREADS_PER_BLOCK 256
#define GPU_MAX_BLOCKS 65536

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      assert(!abort);
      if (abort) exit(code);
   }
}

template<typename dtype, integer shape1, integer shape2>
__global__ void _extract1(const integer n, dtype* phi1, const dtype* phi2, const integer* indices) {
    integer i = threadIdx.x + blockDim.x*blockIdx.x;
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
    integer i = threadIdx.x + blockDim.x*blockIdx.x;
    if (i < n) {
        integer p = indices[i];
        for (integer j = 0; j < shape1; j++) {
            for (integer k = 0; k < shape2; k++) {
                phi1[i*shape1*shape2 + j*shape2 + k] += phi2[p*shape1*shape2 + j*shape2 + k];
            }
        }
    }
}

//template<typename dtype>
//__inline__ __device__ dtype warpReduceSum(dtype val) {
//  for (int offset = warpSize/2; offset > 0; offset /= 2)
//    val += __shfl_down(val, offset);
//  return val;
//}
//
//template<typename dtype>
//__inline__ __device__ void reduceSum(int n, const dtype val, dtype* res) {
//  dtype sum = 0;
//  for(int i = blockIdx.x * blockDim.x + threadIdx.x; 
//      i < n; 
//      i += blockDim.x * gridDim.x) {
//    sum += in[i];
//  }
//  sum = warpReduceSum(sum);
//  if (threadIdx.x & (warpSize - 1) == 0)
//    atomicAdd(out, sum);
//} 
  

template<typename dtype>
__inline__ __device__ void reduceSum(int n, const dtype val, dtype* res) {
    //int x = threadIdx.x + blockDim.x*blockIdx.x; // gridDim.x*blockDim.x*blockIdx.y;
    __shared__ dtype sum[GPU_THREADS_PER_BLOCK];
    sum[threadIdx.x] = val;
    __syncthreads();
    for (int i=blockDim.x/2; i > 0; i/=2) {
        if (threadIdx.x < i)
            sum[threadIdx.x] += sum[threadIdx.x + i];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(&res[0], sum[0]);
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
            gpuErrorCheck(cudaFree(this->data));
            this -> data = NULL;
        }
    }
    void zero() {
        gpuErrorCheck(cudaMemset(this->data, 0, this->size*sizeof(dtype)));
    }
    void copy(integer index, dtype* sdata, integer n) {
        gpuErrorCheck(cudaMemcpy(&(*this)(index), sdata, n*sizeof(dtype), cudaMemcpyDeviceToDevice));
    }
    void extract(const integer index, const integer* indices, const dtype* phiBuf, const integer n) {
        integer blocks = n/GPU_THREADS_PER_BLOCK + 1;
        integer threads = min(GPU_THREADS_PER_BLOCK, n);
        _extract2<dtype, shape1, shape2><<<blocks, threads>>>(n, &(*this)(index), phiBuf, indices);
        gpuErrorCheck(cudaPeekAtLastError());
    }
    void extract(const integer *indices, const dtype* phiBuf, const integer n) {
        integer blocks = n/GPU_THREADS_PER_BLOCK + 1;
        integer threads = min(GPU_THREADS_PER_BLOCK, n);
        _extract1<dtype, shape1, shape2><<<blocks, threads>>>(n, this->data, phiBuf, indices);
        gpuErrorCheck(cudaPeekAtLastError());
        //gpuErrorCheck(cudaDeviceSynchronize());
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
typedef extArrType<scalar, 1> ext_vec;

#endif
#endif
