#pragma once

#include <Eigen/Dense>
#include <cuda_runtime.h>

#include "mean_curvature_types.h"

class CudaRing {
    enum class AngleType { Obtuse, Acute };

    struct CurvatureData {
        double area;
        double cotan;
        AngleType angle_type;
    };

    struct CudaEdge {
        VertexId end;
        VertexId adjs[2];
    };

    struct CurvatureTerm {
        double k = 0.0;
        double w = 0.0;
        __device__ CurvatureTerm() = default;
        __device__ CurvatureTerm(double kv, double wv) : k(kv), w(wv) {}
        __device__ void operator+=(const CurvatureTerm &otr);
        __device__ double Compute();
    };

    VertexId center_;
    CudaEdge edges_[NUM_EDGES];
    int edges_count_ = 0;

  public:
    __host__ __device__ CudaRing() = default;

    __host__ __device__ CudaRing(VertexId center);

    __host__ __device__ int GetCenter();

    __host__ __device__ void SetCenter(int center);

    __host__ __device__ void AddEdge(int end, int adj1, int adj2);

    __device__ double GetSmallestEdgeLength(Eigen::Vector3d *positions);

    /* Not tested yet */
    __device__ void UpdateCenterNormal(Eigen::Vector3d *positions,
                                       Eigen::Vector3d *normals);

    __device__ void ComputeCurvature(Eigen::Vector3d *positions,
                                     Eigen::Vector3d *normals);

  private:
    __device__ CurvatureTerm ComputeEdgeCurvature(int edge_idx,
                                                  Eigen::Vector3d *positions,
                                                  Eigen::Vector3d *normals);

    __device__ CurvatureData GetCurvatureData(VertexId pc, VertexId pl,
                                              VertexId pr,
                                              Eigen::Vector3d *positions);
};
