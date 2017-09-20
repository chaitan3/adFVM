#ifndef GPU_HPP
#define GPU_HPP
#ifdef GPU
#include "common.hpp"

#define GPU_THREADS_PER_BLOCK 256
#define GPU_BLOCKS_PER_GRID 1024
#define GPU_MAX_BLOCKS 65536
#define WARP_SIZE 32

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

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
            assumed = old;
            old = ::atomicCAS(address_as_i, assumed,
                                __float_as_int(::fmaxf(val, __int_as_float(assumed))));
        } while (assumed != old);
    return __int_as_float(old);
}

template<typename dtype>
__inline__ __device__ dtype warpReduceSum(dtype val) {
  for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
    val += __shfl_down(val, offset);
  return val;
}


template<typename dtype>
__inline__ __device__ dtype warpReduceMax(dtype val) {
  for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
    val = max(__shfl_down(val, offset), val);
  return val;
}

template<typename dtype>
__inline__ __device__ void reduceSum(int n, const dtype val, dtype* res) {
  dtype sum = val;
  int i = blockIdx.x * blockDim.x + threadIdx.x; 
  //for(int i = blockIdx.x * blockDim.x + threadIdx.x; 
  //    i < n; 
  //    i += blockDim.x * gridDim.x) {
  //  sum += in[i];
  //}
  int nb = n - (n % WARP_SIZE);
  if (i >= nb) {
      atomicAdd(res, sum);
  } else {
      sum = warpReduceSum<dtype>(sum);
      if (threadIdx.x & (WARP_SIZE - 1) == 0)
        atomicAdd(res, sum);
  }
} 

template<typename dtype>
__inline__ __device__ void reduceMax(int n, const dtype val, dtype* res) {
  dtype sum = val;
  int i = blockIdx.x * blockDim.x + threadIdx.x; 
  //for(int i = blockIdx.x * blockDim.x + threadIdx.x; 
  //    i < n; 
  //    i += blockDim.x * gridDim.x) {
  //  sum += in[i];
  //}
  int nb = n - (n % WARP_SIZE);
  if (i >= nb) {
      atomicMax(res, sum);
  } else {
      sum = warpReduceMax<dtype>(sum);
      if (threadIdx.x & (WARP_SIZE - 1) == 0)
        atomicMax(res, sum);
  }
}
  

//template<typename dtype>
//__inline__ __device__ void reduceSum(int n, const dtype val, dtype* res) {
//    //int x = threadIdx.x + blockDim.x*blockIdx.x; // gridDim.x*blockDim.x*blockIdx.y;
//    __shared__ dtype sum[GPU_THREADS_PER_BLOCK];
//    sum[threadIdx.x] = val;
//    __syncthreads();
//    for (int i=blockDim.x/2; i > 0; i/=2) {
//        if (threadIdx.x < i)
//            sum[threadIdx.x] += sum[threadIdx.x + i];
//        __syncthreads();
//    }
//    if (threadIdx.x == 0)
//        atomicAdd(&res[0], sum[0]);
//}

template <typename dtype, integer shape1=1, integer shape2=1, integer shape3=1>
class gpuArrType: public arrType<dtype, shape1, shape2, shape3> {
    public:
    void alloc() {
        gpuErrorCheck(cudaMalloc(&this->data, this->bufSize));
    }
    void dealloc() {
        gpuErrorCheck(cudaFree(this->data));
    }
    gpuArrType(const integer shape, bool zero=false, bool keepMemory=false, int id=-1):
    arrType<dtype, shape1, shape2, shape3>(shape, zero, keepMemory, id) {}

    gpuArrType(const integer shape, dtype* data):
    arrType<dtype, shape1, shape2, shape3>(shape, data) {}

    void zero() {
        gpuErrorCheck(cudaMemset(this->data, 0, this->bufSize));
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
        cout << "transferring to host: " << this->bufSize << endl;
        gpuErrorCheck(cudaMemcpy(hdata, this->data, this->bufSize, cudaMemcpyDeviceToHost));
        return hdata;
    }
    void toDevice(dtype* data) {
        cout << "transferring to device: " << this->bufSize << endl;
        gpuErrorCheck(cudaMemcpy(this->data, data, this->bufSize, cudaMemcpyHostToDevice));
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
