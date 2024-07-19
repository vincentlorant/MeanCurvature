#include <Eigen/Dense>

#include "common/cuda/cuda_helpers.h"
#include "common/types.h"
#include "cuda/curvature.h"
#include "cuda/ring_manip.h"

__global__ void ComputeEdgesLengthKernel(
  CudaRing* rings, int n, Eigen::Vector3d* positions, double* out) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    out[i] = CudaRingManip(&rings[i]).GetSmallestEdgeLength(positions);
  }
}

std::vector<double> ComputeEdgesLength(
  CudaRing* d_cuda_rings, size_t rings_n, Eigen::Vector3d* d_vertices) {
  dim3 block_size(128);
  dim3 grid_size((rings_n + block_size.x - 1) / block_size.x);

  double* d_small_values;
  size_t values_mem_size = sizeof(double) * rings_n;
  cuCheck(cudaMalloc(&d_small_values, values_mem_size));

  ComputeEdgesLengthKernel<<<grid_size, block_size>>>(
    d_cuda_rings, rings_n, d_vertices, d_small_values);

  CheckKernelAndSync();

  std::vector<double> smalls(rings_n, 0);
  cudaMemcpy(smalls.data(), d_small_values, values_mem_size,
    cudaMemcpyKind::cudaMemcpyDeviceToHost);
  cuCheck(cudaFree(d_small_values));

  return smalls;
}

__global__ void
ScalePositionsKernel(Eigen::Vector3d* positions, int n, float amount) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    positions[i] *= amount;
  }
}

void ScalePositions(
  Eigen::Vector3d* d_vertices, size_t vertices_n, float scalar_fix) {
  dim3 block_size(128);
  dim3 grid_size((vertices_n + block_size.x - 1) / block_size.x);
  ScalePositionsKernel<<<grid_size, block_size>>>(
    d_vertices, vertices_n, scalar_fix);
  CheckKernelAndSync();
}

__global__ void OptimizeMeanCurvatureKernel(CudaRing* rings,
  int n,
  Eigen::Vector3d* positions,
  Eigen::Vector3d* normals) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    CudaRingManip(&rings[i]).ComputeCurvature(positions, normals);
  }
}

void OptimizeMeanCurvature(CudaRing* d_cuda_rings,
  int vertices_n,
  Eigen::Vector3d* d_vertices,
  Eigen::Vector3d* d_normals) {
  dim3 block_size(128);
  dim3 grid_size((vertices_n + block_size.x - 1) / block_size.x);
  OptimizeMeanCurvatureKernel<<<grid_size, block_size>>>(
    d_cuda_rings, vertices_n, d_vertices, d_normals);

  CheckKernelAndSync();
}