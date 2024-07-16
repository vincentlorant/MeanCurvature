#include "mean_curvature.h"

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <open3d/Open3D.h>

#include <argparse/argparse.hpp>
#include <cassert>

#include "common/cuda/cuda_helpers.h"
#include "common/cuda_ring.h"
#include "common/helpers.h"
#include "common/host_ring.h"
#include "common/types.h"
#include "cuda/curvature.h"

void MeanCurvature::operator()(MeshPtr mesh, int nitr, double scale_factor) {
  // check inputs
  if (mesh == nullptr) {
    LOG(FATAL) << "Invalid mesh";
  }
  if (nitr <= 0) {
    LOG(FATAL) << "Invalid nitr param:" << nitr;
  }
  if (scale_factor <= 0) {
    LOG(FATAL) << "Invalid scale_factor param:" << scale_factor;
  }

  // get device data
  auto [d_cuda_rings, d_edges] = GetCudaRingsAndEdges(mesh);
  auto [d_vertices, d_normals, mem_size] = CopyMeshDataToDevice(mesh);
  size_t vertices_n = mesh->vertices_.size();

  auto edges_lengths = ComputeEdgesLength(d_cuda_rings, vertices_n, d_vertices);
  double smallest_edge_length =
      *std::min_element(edges_lengths.begin(), edges_lengths.end());

  // scale mesh up to ease computation
  double scalar_fix = scale_factor / smallest_edge_length;
  ScalePositions(d_vertices, vertices_n, scalar_fix);

  // Optimization
  auto f = [&](auto d_cuda_rings, auto d_vertices, auto d_normals) {
    for (int i = 0; i < nitr; i++) {
      OptimizeMeanCurvature(d_cuda_rings, vertices_n, d_vertices, d_normals);
    }
  };
  TimeCudaEvent(f(d_cuda_rings, d_vertices, d_normals), nitr);

  // Scale back vertices to original values
  ScalePositions(d_vertices, vertices_n, 1.0 / scalar_fix);

  // Get back updated vertices to host
  cuCheck(cudaMemcpy(mesh->vertices_.data(), d_vertices, mem_size,
                     cudaMemcpyKind::cudaMemcpyDeviceToHost));

  //
  cuCheck(cudaFree(d_cuda_rings));
  cuCheck(cudaFree(d_edges));
  cuCheck(cudaFree(d_vertices));
  cuCheck(cudaFree(d_normals));

  // Update normals with new vertices
  mesh->ComputeVertexNormals();
}

std::tuple<Eigen::Vector3d *, Eigen::Vector3d *, size_t>
MeanCurvature::CopyMeshDataToDevice(
    std::shared_ptr<open3d::geometry::TriangleMesh> mesh) {
  size_t vertices_num = mesh->vertices_.size();
  Eigen::Vector3d *d_vertices;
  Eigen::Vector3d *d_normals;
  size_t vec_mem_size = sizeof(Eigen::Vector3d) * vertices_num;
  cuCheck(cudaMalloc(&d_vertices, vec_mem_size));
  cuCheck(cudaMalloc(&d_normals, vec_mem_size));
  cuCheck(cudaMemcpy(d_vertices, mesh->vertices_.data(), vec_mem_size,
                     cudaMemcpyKind::cudaMemcpyHostToDevice));
  cuCheck(cudaMemcpy(d_normals, mesh->vertex_normals_.data(), vec_mem_size,
                     cudaMemcpyKind::cudaMemcpyHostToDevice));

  return {d_vertices, d_normals, vec_mem_size};
}
