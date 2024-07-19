#pragma once

#include <cuda_runtime.h>

#include <Eigen/Dense>

#include "common/types.h"

class CudaRingManip {
 private:
  enum class AngleType { Obtuse, Acute };

  struct CurvatureData {
    double area;
    double cotan;
    AngleType angle_type;
  };

  struct CurvatureTerm {
    double k = 0.0;
    double w = 0.0;
    __device__ CurvatureTerm();
    __device__ CurvatureTerm(double kv, double wv);
    __device__ void operator+=(const CurvatureTerm &otr);
    __device__ double NormalizeCurvature();
  };

  VertexId center_;
  CudaEdge *edges_;
  int edges_count_ = 0;

 public:
  __device__ CudaRingManip(VertexId center, CudaEdge *edges, int edges_count);

  __device__ CudaRingManip(CudaRing *ring);

  __device__ ~CudaRingManip();

  __device__ void CleanEdges();

  __device__ int GetCenter();

  __device__ void SetCenter(int center);

  __device__ void AddEdge(int end, int adj1, int adj2);

  __device__ double GetSmallestEdgeLength(Eigen::Vector3d *positions);

  __device__ void
  ComputeCurvature(Eigen::Vector3d *positions, Eigen::Vector3d *normals);

 private:
  __device__ CurvatureTerm ComputeEdgeCurvature(
    int edge_idx, Eigen::Vector3d *positions, Eigen::Vector3d *normals);

  __device__ CurvatureData GetCurvatureData(const Eigen::Vector3d &pe,
    const Eigen::Vector3d &pa,
    const Eigen::Vector3d &pb);
};
