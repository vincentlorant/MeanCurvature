#pragma once

#include <cuda_runtime.h>

#include <iostream>

#define cuCheck(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

#define cuAsyncCheck() \
  cudaError_t cerror;  \
  cudaCheckLast(cerror);

#define cuCheckLast(cerror)                            \
  cerror = cudaGetLastError();                         \
  if (cerror != cudaSuccess) {                         \
    printf("Error: %s\n", cudaGetErrorString(cerror)); \
  }

#define CheckKernelAndSync()   \
  cuCheck(cudaGetLastError()); \
  cuCheck(cudaDeviceSynchronize());

#define TimeCudaEvent(func, itr)            \
  cudaEvent_t start, stop;                  \
  cudaEventCreate(&start);                  \
  cudaEventCreate(&stop);                   \
  cudaEventRecord(start);                   \
  func;                                     \
  cudaEventRecord(stop);                    \
  cudaEventSynchronize(stop);               \
  float time;                               \
  cudaEventElapsedTime(&time, start, stop); \
  cudaEventDestroy(start);                  \
  cudaEventDestroy(stop);                   \
  LOG(INFO) << "kernel avg time:" << (time / itr) << "ms";
