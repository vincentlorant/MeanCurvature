#pragma once

#include <Eigen/Dense>

#include "common/types.h"

template <typename T>
__host__ T GetSmallestEdgeLength(CudaRing *d_cuda_rings, size_t nrings,
                                 Eigen::Matrix<T, 3, 1> *d_vertices,
                                 size_t vertices_num);

template <typename T>
__host__ void AddAmount(Eigen::Matrix<T, 3, 1> *d_vertices, size_t vertices_num,
                        T amount);

template <typename T>
__host__ void UpdateNormals(CudaRing *d_rings,
                            Eigen::Matrix<T, 3, 1> *d_positions,
                            Eigen::Matrix<T, 3, 1> *d_normals, size_t n);

template <typename T>
__host__ void MeanCurvatureOptimization(CudaRing *d_rings,
                                        Eigen::Matrix<T, 3, 1> *d_positions,
                                        Eigen::Matrix<T, 3, 1> *d_normals,
                                        size_t n);