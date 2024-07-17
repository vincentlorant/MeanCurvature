#include "mean_curvature.h"

#include <glog/logging.h>
#include <oneapi/tbb.h>
#include <open3d/Open3D.h>

#include <Eigen/Dense>
#include <type_traits>

#include "common/cuda/cuda_helpers.h"
#include "common/cuda_ring.h"
#include "common/host_ring.h"
#include "common/types.h"
#include "cuda/curvature.h"

/*
 * When the mean-curvature is processed with double floating point precision,
 * vertices and normals can be directly copied to device allocated arrays
 */
template <typename T, typename = std::enable_if_t<std::is_same_v<T, double>>>
auto CopyMeshDataToDevice(std::shared_ptr<open3d::geometry::TriangleMesh> mesh)
  -> std::tuple<Eigen::Vector3d *, Eigen::Vector3d *, size_t> {
  size_t n = mesh->vertices_.size();
  size_t vec_mem_size = sizeof(Eigen::Vector3d) * n;
  Eigen::Vector3d *d_vertices;
  Eigen::Vector3d *d_normals;
  cuCheck(cudaMalloc(&d_vertices, vec_mem_size));
  cuCheck(cudaMalloc(&d_normals, vec_mem_size));
  cuCheck(cudaMemcpy(d_vertices, mesh->vertices_.data(), vec_mem_size,
    cudaMemcpyKind::cudaMemcpyHostToDevice));
  cuCheck(cudaMemcpy(d_normals, mesh->vertex_normals_.data(), vec_mem_size,
    cudaMemcpyKind::cudaMemcpyHostToDevice));

  return {d_vertices, d_normals, vec_mem_size};
}

/*
 * When mean-curvature is processed with single floating points precision,
 * vertices and normals need to be converted before to be copied to device
 * allocated arrays
 */

auto MapDtoF(
  std::vector<Eigen::Vector3d> &input) -> std::vector<Eigen::Vector3f> {
  size_t n = input.size();
  Eigen::Map<Eigen::Matrix3Xd> v(input[0].data(), 3, n);

  return {v.cast<float>().colwise().begin(), v.cast<float>().colwise().end()};
}

template <typename T, typename = std::enable_if_t<std::is_same_v<T, float>>>
auto CopyMeshDataToDevice(std::shared_ptr<open3d::geometry::TriangleMesh> mesh)
  -> std::tuple<Eigen::Vector3f *, Eigen::Vector3f *, size_t> {
  size_t n = mesh->vertices_.size();
  Eigen::Vector3f *d_vertices;
  Eigen::Vector3f *d_normals;
  size_t vec_mem_size = sizeof(Eigen::Vector3f) * n;
  cuCheck(cudaMalloc(&d_vertices, vec_mem_size));
  cuCheck(cudaMalloc(&d_normals, vec_mem_size));
  cuCheck(cudaMemcpy(d_vertices, MapDtoF(mesh->vertices_).data(), vec_mem_size,
    cudaMemcpyKind::cudaMemcpyHostToDevice));
  cuCheck(cudaMemcpy(d_normals, MapDtoF(mesh->vertex_normals_).data(),
    vec_mem_size, cudaMemcpyKind::cudaMemcpyHostToDevice));

  return {d_vertices, d_normals, vec_mem_size};
}

/*
 * Copy device array of double to host
 */
auto CopyDataToHost(
  Eigen::Vector3d *d_arr, size_t n) -> std::vector<Eigen::Vector3d> {
  std::vector<typename Eigen::Vector3d> v(n);
  size_t mem_size = sizeof(Eigen::Vector3d) * n;
  cuCheck(cudaMemcpy(
    v.data(), d_arr, mem_size, cudaMemcpyKind::cudaMemcpyDeviceToHost));
  return v;
}

/*
 * Copy device array of float to host and convert it to double
 */
auto CopyDataToHost(
  Eigen::Vector3f *d_arr, size_t n) -> std::vector<Eigen::Vector3d> {
  std::vector<typename Eigen::Vector3f> v(n);
  size_t mem_size = sizeof(Eigen::Vector3f) * n;
  cuCheck(cudaMemcpy(
    v.data(), d_arr, mem_size, cudaMemcpyKind::cudaMemcpyDeviceToHost));
  Eigen::Map<Eigen::Matrix3Xf> vd(v[0].data(), 3, v.size());

  return {
    vd.cast<double>().colwise().begin(), vd.cast<double>().colwise().end()};
}

template <typename T>
void MeanCurvature<T>::operator()(
  std::shared_ptr<open3d::geometry::TriangleMesh> mesh,
  int nitr /* = 50 */,
  float scale_factor /* = 0.035 */) {
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

  // // get device data
  auto [d_cuda_rings, d_edges] = GetCudaRingsAndEdges(mesh);
  auto [d_vertices, d_normals, mem_size] = CopyMeshDataToDevice<T>(mesh);

  // Cuda kernel config
  size_t verices_num = mesh->vertices_.size();

  // // pre optimization
  T smallest_edge_length = GetSmallestEdgeLength<T>(
    d_cuda_rings, verices_num, d_vertices, verices_num);
  T scalar_fix = scale_factor / smallest_edge_length;
  AddAmount<T>(d_vertices, verices_num, scalar_fix);

  // optimization
  auto f = [&](auto d_cuda_rings, auto d_vertices, auto d_normals) {
    for (int i = 0; i < nitr; i++) {
      MeanCurvatureOptimization<T>(
        d_cuda_rings, d_vertices, d_normals, verices_num);
    }
  };
  TimeCudaEvent(f(d_cuda_rings, d_vertices, d_normals), nitr);

  // post optimization
  UpdateNormals<T>(d_cuda_rings, d_vertices, d_normals, verices_num);

  AddAmount<T>(d_vertices, verices_num, 1.0 / scalar_fix);

  // update mesh with optimized data
  cuCheck(cudaDeviceSynchronize());
  mesh->vertices_ = CopyDataToHost(d_vertices, verices_num);
  mesh->vertex_normals_ = CopyDataToHost(d_normals, verices_num);

  // clean allocated device memory
  cuCheck(cudaFree(d_cuda_rings));
  cuCheck(cudaFree(d_edges));
  cuCheck(cudaFree(d_vertices));
  cuCheck(cudaFree(d_normals));

  // update normals
  tbb::parallel_for(size_t(0), mesh->triangles_.size(), [&](size_t i) {
    auto &t = mesh->triangles_[i];
    mesh->triangle_normals_[i] = mesh->vertex_normals_[t(0)] +
                                 mesh->vertex_normals_[t(1)] +
                                 mesh->vertex_normals_[t(2)];
  });
}

template class MeanCurvature<double>;
template class MeanCurvature<float>;