
#include <Eigen/Dense>

#include "common/types.h"

std::vector<double> ComputeEdgesLength(CudaRing* d_cuda_rings, size_t rings_n,
                                       Eigen::Vector3d* d_vertices);

void ScalePositions(Eigen::Vector3d* d_vertices, size_t vertices_n,
                    float scalar_fix);

void OptimizeMeanCurvature(CudaRing* d_cuda_rings, int vertices_num,
                           Eigen::Vector3d* d_vertices,
                           Eigen::Vector3d* d_normals);