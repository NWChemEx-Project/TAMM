#pragma once

#include <assert.h>
#include <stdio.h>
#include <string>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
#include "tamm/gpuStreams.hpp"
#endif

#ifdef USE_TALSH
#define USE_TALSH_T
#endif
#undef USE_TALSH

#ifdef USE_CUDA
#define CHECK_ERR(x)                           \
  {                                            \
    cudaError_t err = cudaGetLastError();      \
    if(cudaSuccess != err) {                   \
      printf("%s\n", cudaGetErrorString(err)); \
      exit(1);                                 \
    }                                          \
  }

#define CUDA_SAFE(x)                                                                \
  if(cudaSuccess != (x)) {                                                          \
    printf("CUDA API FAILED AT LINE %d OF FILE %s errorcode: %s, %s\n", __LINE__, __FILE__, \
           cudaGetErrorName(x), cudaGetErrorString(cudaGetLastError())); \
    exit(100);                                                          \
  }
#endif // USE_CUDA

#ifdef USE_HIP
#define CHECK_ERR(x)                          \
  {                                           \
    hipError_t err = hipGetLastError();       \
    if(hipSuccess != err) {                   \
      printf("%s\n", hipGetErrorString(err)); \
      exit(1);                                \
    }                                         \
  }

#define HIP_SAFE(x)                                                                \
  if(hipSuccess != (x)) {                                                          \
    printf("HIP API FAILED AT LINE %d OF FILE %s errorcode: %s, %s\n", __LINE__, __FILE__, \
           hipGetErrorName(x), hipGetErrorString(hipGetLastError()));    \
    exit(100);                                                                     \
  }
#endif // USE_HIP

typedef long Integer;
// static int notset;

#define DIV_UB(x, y) ((x) / (y) + ((x) % (y) ? 1 : 0))
#define TG_MIN(x, y) ((x) < (y) ? (x) : (y))

void        initMemModule();
std::string check_memory_req(const int nDevices, const int cc_t_ts, const int nbf);

void* getGpuMem(size_t bytes);
void* getHostMem(size_t bytes);
void  freeHostMem(void* p);
void  freeGpuMem(void* p);

void finalizeMemModule();

struct hostEnergyReduceData_t {
  double* result_energy;
  double* host_energies;
  size_t  num_blocks;
  double  factor;
};
