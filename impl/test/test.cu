#include <iostream>
#include <vector>

#include "cuda_helpers.h"
#include "test.h"

template <typename T>
__global__ void add_kernel(T* d_A, T value, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    d_A[i] += value;
  }
}

void add(std::vector<float>& arr) {
  std::cout << "hello:" << arr.size() << std::endl;
  float* d_A;
  size_t mem_size = arr.size() * sizeof(float);

  cuCheck(cudaMalloc(&d_A, mem_size));
  cuCheck(cudaMemcpy(d_A, arr.data(), mem_size,
                     cudaMemcpyKind::cudaMemcpyHostToDevice));

  int thread_per_block = 256;
  int num_blocks = (arr.size() + thread_per_block - 1) / thread_per_block;
  add_kernel<float><<<num_blocks, thread_per_block>>>(d_A, 5.0, arr.size());
  cuCheck(cudaDeviceSynchronize());

  cuCheck(cudaMemcpy(arr.data(), d_A, mem_size,
                     cudaMemcpyKind::cudaMemcpyDeviceToHost));

  cuCheck(cudaFree(d_A));

  cuCheck(cudaDeviceSynchronize());
}