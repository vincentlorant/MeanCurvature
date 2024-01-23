#pragma once

#include <Eigen/Dense>
#include <cuda_runtime.h>

#include "mean_curvature_types.h"

struct CudaEdge {
    VertexId end;
    VertexId adjs[2];
};

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

    struct CurvatureTerm {
        T k = 0.0;
        T w = 0.0;
        __device__ CurvatureTerm() = default;
        __device__ CurvatureTerm(T kv, T wv) : k(kv), w(wv) {}
        __device__ void operator+=(const CurvatureTerm &otr);
        __device__ T NormalizeCurvature();
    };

    VertexId center_;
    CudaEdge *edges_;
    int edges_count_ = 0;

  public:
    __host__ __device__ CudaRing(VertexId center, CudaEdge *edges,
                                 int edges_count);

    __host__ __device__ ~CudaRing();

    __host__ __device__ void CleanEdges() { cudaFree(edges_); }

    __host__ __device__ int GetCenter();

    __host__ __device__ void SetCenter(int center);

    __host__ __device__ void AddEdge(int end, int adj1, int adj2);

    __device__ T GetSmallestEdgeLength(Vector *positions);

    __device__ void ComputeCurvature(Vector *positions, Vector *normals);

  private:
    __device__ CurvatureTerm ComputeEdgeCurvature(int edge_idx,
                                                  Vector *positions,
                                                  Vector *normals);

    __device__ CurvatureData GetCurvatureData(const Vector &pe,
                                              const Vector &pa,
                                              const Vector &pb);
};

extern template class CudaRing<double>;
extern template class CudaRing<float>;