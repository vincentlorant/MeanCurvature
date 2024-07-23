#include <Eigen/Dense>
#include <algorithm>
#include <vector>

#include "common/cuda/cuda_helpers.h"
#include "common/types.h"
#include "cuda/curvature.h"
#include "cuda/ring_manip.h"

// AddAmount

template <typename T>
__global__ void
AddAmountKernel(Eigen::Matrix<T, 3, 1> *arr, size_t n, T amount) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    arr[i] *= amount;
  }
}

template <typename T>
__host__ void
AddAmount(Eigen::Matrix<T, 3, 1> *d_vertices, size_t vertices_num, T amount) {
  dim3 block_size(128);
  dim3 grid_size((vertices_num + block_size.x - 1) / block_size.x);

  AddAmountKernel<T>
    <<<grid_size, block_size>>>(d_vertices, vertices_num, amount);
  CheckKernelAndSync();
}

template void AddAmount(
  Eigen::Matrix<double, 3, 1> *d_vertices, size_t vertices_num, double amount);
template void AddAmount(
  Eigen::Matrix<float, 3, 1> *d_vertices, size_t vertices_num, float amount);

// GetSmallestEdgeLength

template <typename T>
__global__ void GetSmallestEdgesLengthsKernel(
  CudaRing *d_rings, int n, Eigen::Matrix<T, 3, 1> *positions, T *out) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    out[i] = CudaRingManip<T>(&d_rings[i]).GetSmallestEdgeLength(positions);
  }
}

template <typename T>
__host__ T GetSmallestEdgeLength(CudaRing *d_rings,
  size_t nrings,
  Eigen::Matrix<T, 3, 1> *d_vertices,
  size_t vertices_num) {
  dim3 block_size(128);
  dim3 grid_size((vertices_num + block_size.x - 1) / block_size.x);

  T *d_small_values;
  size_t values_mem_size = sizeof(T) * vertices_num;
  cuCheck(cudaMalloc(&d_small_values, values_mem_size));

  GetSmallestEdgesLengthsKernel<T>
    <<<grid_size, block_size>>>(d_rings, nrings, d_vertices, d_small_values);

  CheckKernelAndSync();

  std::vector<T> smalls(vertices_num, 0.0);
  cudaMemcpy(smalls.data(), d_small_values, values_mem_size,
    cudaMemcpyKind::cudaMemcpyDeviceToHost);
  cuCheck(cudaFree(d_small_values));

  return *std::min_element(smalls.begin(), smalls.end());
}

template double GetSmallestEdgeLength(CudaRing *d_rings,
  size_t nrings,
  Eigen::Matrix<double, 3, 1> *d_vertices,
  size_t vertices_num);
template float GetSmallestEdgeLength(CudaRing *d_rings,
  size_t nrings,
  Eigen::Matrix<float, 3, 1> *d_vertices,
  size_t vertices_num);

// UpdateNormals

template <typename T>
__global__ void UpdateNormalsKernel(CudaRing *d_rings,
  Eigen::Matrix<T, 3, 1> *d_positions,
  Eigen::Matrix<T, 3, 1> *d_normals,
  size_t n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    CudaRingManip<T>(&d_rings[i]).UpdateCenterNormal(d_positions, d_normals);
  }
}

template <typename T>
__host__ void UpdateNormals(CudaRing *d_rings,
  Eigen::Matrix<T, 3, 1> *d_positions,
  Eigen::Matrix<T, 3, 1> *d_normals,
  size_t n) {
  dim3 block_size(128);
  dim3 grid_size((n + block_size.x - 1) / block_size.x);

  UpdateNormalsKernel<T>
    <<<grid_size, block_size>>>(d_rings, d_positions, d_normals, n);

  CheckKernelAndSync();
}

template void UpdateNormals(CudaRing *rings,
  Eigen::Matrix<double, 3, 1> *d_positions,
  Eigen::Matrix<double, 3, 1> *d_normals,
  size_t n);
template void UpdateNormals(CudaRing *rings,
  Eigen::Matrix<float, 3, 1> *d_positions,
  Eigen::Matrix<float, 3, 1> *d_normals,
  size_t n);

// MeanCurvature

template <typename T>
__global__ void MeanCurvatureKernel(CudaRing *d_rings,
  Eigen::Matrix<T, 3, 1> *d_positions,
  Eigen::Matrix<T, 3, 1> *d_normals,
  size_t n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    CudaRingManip<T>(&d_rings[i]).ComputeCurvature(d_positions, d_normals);
  }
}

template <typename T>
__host__ void MeanCurvatureOptimization(CudaRing *d_rings,
  Eigen::Matrix<T, 3, 1> *d_positions,
  Eigen::Matrix<T, 3, 1> *d_normals,
  size_t n) {
  dim3 block_size(128);
  dim3 grid_size((n + block_size.x - 1) / block_size.x);

  MeanCurvatureKernel<T>
    <<<grid_size, block_size>>>(d_rings, d_positions, d_normals, n);

  CheckKernelAndSync();
}

template void MeanCurvatureOptimization(CudaRing *d_rings,
  Eigen::Matrix<double, 3, 1> *d_positions,
  Eigen::Matrix<double, 3, 1> *d_normals,
  size_t n);

template void MeanCurvatureOptimization(CudaRing *d_rings,
  Eigen::Matrix<float, 3, 1> *d_positions,
  Eigen::Matrix<float, 3, 1> *d_normals,
  size_t n);