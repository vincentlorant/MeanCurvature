#pragma once

#include <Eigen/Dense>
#include <cuda_runtime.h>

#include "types.h"

struct CudaEdge {
    VertexId end;
    VertexId adjs[2];
};

/// \class CudaRing
///
/// \brief CudaRing represents a vertex and its contour vertices
template <typename T> class CudaRing {
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

  public:
    __host__ __device__ CudaRing() = delete;

    __host__ __device__ CudaRing(VertexId center, CudaEdge *edges,
                                 int edges_count);

    __host__ __device__ ~CudaRing();

  public:
    __device__ int GetCenter() const;

    __device__ T GetSmallestEdgeLength(Vector *positions) const;

    __device__ void ComputeCurvature(Vector *positions, Vector *normals) const;

  private:
    __device__ Curvature ComputeEdgeCurvature(int edge_idx, Vector *positions,
                                              Vector *normals) const;

    __device__ CurvatureData ComputeCurvatureWeight(const Vector &pe,
                                                    const Vector &pa,
                                                    const Vector &pb) const;

    __device__ Vector Normalized(const Vector &v) const;

    __device__ T Norm(const Vector &v) const;

    VertexId center_;
    CudaEdge *edges_;
    int edges_count_ = 0;
};

extern template class CudaRing<double>;
extern template class CudaRing<float>;