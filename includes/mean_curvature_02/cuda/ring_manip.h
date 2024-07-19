#pragma once

#include <cuda_runtime.h>

#include <Eigen/Dense>

#include "common/types.h"

template <typename T>
class CudaRingManip {
 public:
  using Vector = Eigen::Matrix<T, 3, 1>;

 private:
  enum class AngleType { Obtuse, Acute };

  struct CurvatureData {
    T area;
    T cotan;
    AngleType angle_type;
  };

  struct Curvature {
    T k = 0.0;
    T w = 0.0;
    __device__ Curvature() = default;
    __device__ Curvature(T kv, T wv) : k(kv), w(wv) {}
    __device__ void operator+=(const Curvature &otr);
    __device__ T NormalizeCurvature();
  };

 private:
  VertexId center_;
  CudaEdge *edges_;
  int edges_count_ = 0;

 public:
  __device__ CudaRingManip() = delete;

  __device__ CudaRingManip(VertexId center, CudaEdge *edges, int edges_count);

  __device__ CudaRingManip(CudaRing *ring);

  __device__ ~CudaRingManip();

 public:
  __device__ int GetCenter() const;

  __device__ T GetSmallestEdgeLength(Vector *positions) const;

  __device__ void ComputeCurvature(Vector *positions, Vector *normals) const;

  __device__ void UpdateCenterNormal(Vector *positions, Vector *normals) const;

 private:
  __device__ Curvature ComputeEdgeCurvature(
    int edge_idx, Vector *positions, const Vector &center_normal) const;

  __device__ CurvatureData ComputeCurvatureWeight(
    const Vector &pc, const Vector &pe, const Vector &px) const;

  __device__ Vector Normalized(const Vector &v) const;

  __device__ T Norm(const Vector &v) const;
};

extern template class CudaRingManip<double>;
extern template class CudaRingManip<float>;