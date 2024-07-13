#include "mean_curvature.h"

#include <Eigen/Dense>
#include <memory>
#include <tuple>
#include <type_traits>

#include <oneapi/tbb.h>
#include <open3d/Open3D.h>

#include "cuda_helpers.h"
#include "cuda_ring.h"
#include "host_ring.h"

template <typename T>
__global__ void MeanCurvature(CudaRing<T> *rings, int n,
                              typename CudaRing<T>::Vector *positions,
                              typename CudaRing<T>::Vector *normals) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        rings[i].ComputeCurvature(positions, normals);
    }
}

template <typename T>
__global__ void UpdateNormals(CudaRing<T> *rings, int n,
                              typename CudaRing<T>::Vector *positions,
                              typename CudaRing<T>::Vector *normals) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        rings[i].UpdateCenterNormal(positions, normals);
    }
}

template <typename T>
__global__ void AddAmount(typename CudaRing<T>::Vector *arr, int n,
                          float amount) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        arr[i] *= amount;
    }
}

/*
 * When the mean-curvature is processed with double floating point precision,
 * vertices and normals can be directly copied to device allocated arrays
 */
template <typename T = double, typename = typename std::enable_if<
                                   std::is_same<T, double>::value>::type>
std::tuple<Eigen::Vector3d *, Eigen::Vector3d *, size_t>
CopyMeshDataToDevice(std::shared_ptr<open3d::geometry::TriangleMesh> mesh) {
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

std::vector<Eigen::Vector3f> MapDtoF(std::vector<Eigen::Vector3d> &input) {
    size_t n = input.size();
    Eigen::Map<Eigen::Matrix3Xd> v(input[0].data(), 3, n);

    return std::vector<Eigen::Vector3f>(v.cast<float>().colwise().begin(),
                                        v.cast<float>().colwise().end());
}

template <typename T = float, typename = typename std::enable_if<
                                  std::is_same<T, float>::value>::type>
std::tuple<Eigen::Vector3f *, Eigen::Vector3f *, size_t>
CopyMeshDataToDevice(std::shared_ptr<open3d::geometry::TriangleMesh> mesh) {
    size_t n = mesh->vertices_.size();
    Eigen::Vector3f *d_vertices;
    Eigen::Vector3f *d_normals;
    size_t vec_mem_size = sizeof(Eigen::Vector3f) * n;
    cuCheck(cudaMalloc(&d_vertices, vec_mem_size));
    cuCheck(cudaMalloc(&d_normals, vec_mem_size));
    cuCheck(cudaMemcpy(d_vertices, MapDtoF(mesh->vertices_).data(),
                       vec_mem_size, cudaMemcpyKind::cudaMemcpyHostToDevice));
    cuCheck(cudaMemcpy(d_normals, MapDtoF(mesh->vertex_normals_).data(),
                       vec_mem_size, cudaMemcpyKind::cudaMemcpyHostToDevice));

    return {d_vertices, d_normals, vec_mem_size};
}

template <typename T>
CudaRing<T> ToCudaRing(const HostRing &ring, CudaEdge *d_edges) {

    return CudaRing<T>(ring.GetCenter(), d_edges, ring.GetCompleteEdgeCount());
}

CudaEdge *CreateCudaEdges(const HostRingsMap &host_rings) {
    size_t valid_edges_count = 0;
    for (auto &r : host_rings) {
        valid_edges_count += r.second.GetCompleteEdgeCount();
    }
    std::vector<CudaEdge> edges;
    edges.reserve(valid_edges_count);
    for (const auto &r : host_rings) {
        for (const auto &e : r.second.GetEdges()) {
            if (e.second.Complete()) {
                edges.push_back({e.first, e.second.adjs[0], e.second.adjs[1]});
            }
        }
    }
    CudaEdge *d_edges;
    size_t mem_size = sizeof(CudaEdge) * valid_edges_count;
    cuCheck(cudaMalloc(&d_edges, mem_size));
    cuCheck(cudaMemcpy(d_edges, edges.data(), mem_size,
                       cudaMemcpyKind::cudaMemcpyHostToDevice));

    return d_edges;
}

template <typename T>
CudaRing<T> *CreateCudaRings(const HostRingsMap &host_rings,
                             CudaEdge *d_edges) {
    CudaEdge *pos = d_edges;
    size_t nrings = host_rings.size();
    std::vector<CudaRing<T>> cuda_rings;
    cuda_rings.reserve(nrings);
    for (const auto &r : host_rings) {
        cuda_rings.push_back(ToCudaRing<T>(r.second, pos));
        pos += r.second.GetCompleteEdgeCount();
    }
    CudaRing<T> *d_cuda_rings;
    size_t ring_mem_size = (sizeof(CudaRing<T>) * nrings);
    cuCheck(cudaMalloc(&d_cuda_rings, ring_mem_size));
    cuCheck(cudaMemcpy(d_cuda_rings, cuda_rings.data(), ring_mem_size,
                       cudaMemcpyKind::cudaMemcpyHostToDevice));

    return d_cuda_rings;
}

template <typename T>
std::pair<CudaRing<T> *, CudaEdge *>
GetCudaRingsAndEdges(std::shared_ptr<open3d::geometry::TriangleMesh> mesh) {
    auto host_rings = CreateHostRings(mesh);
    CudaEdge *d_edges = CreateCudaEdges(host_rings);
    CudaRing<T> *d_cuda_rings = CreateCudaRings<T>(host_rings, d_edges);

    return {d_cuda_rings, d_edges};
}

/*
 * Copy device array of double to host
 */
std::vector<Eigen::Vector3d> CopyDataToHost(Eigen::Vector3d *d_arr, size_t n) {
    std::vector<typename Eigen::Vector3d> v(n);
    size_t mem_size = sizeof(Eigen::Vector3d) * n;
    cuCheck(cudaMemcpy(v.data(), d_arr, mem_size,
                       cudaMemcpyKind::cudaMemcpyDeviceToHost));
    return v;
}

/*
 * Copy device array of float to host and convert it to double
 */
std::vector<Eigen::Vector3d> CopyDataToHost(Eigen::Vector3f *d_arr, size_t n) {
    std::vector<typename Eigen::Vector3f> v(n);
    size_t mem_size = sizeof(Eigen::Vector3f) * n;
    cuCheck(cudaMemcpy(v.data(), d_arr, mem_size,
                       cudaMemcpyKind::cudaMemcpyDeviceToHost));
    Eigen::Map<Eigen::Matrix3Xf> vd(v[0].data(), 3, v.size());

    return std::vector<Eigen::Vector3d>(vd.cast<double>().colwise().begin(),
                                        vd.cast<double>().colwise().end());
}

template <typename T>
__global__ void GetSmallestEdgesLengths(CudaRing<T> *rings, int n,
                                        typename CudaRing<T>::Vector *positions,
                                        T *out) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        out[i] = rings[i].GetSmallestEdgeLength(positions);
    }
}

/*
 * Uniformly scales up vertices based on the smallest edge's length to
 * ensure computation-friendly values, especially when dealing with
 * small edge lengths.
 */
template <typename T>
T GetSmallestEdgeLength(CudaRing<T> *d_cuda_rings, size_t nrings,
                        typename CudaRing<T>::Vector *d_vertices,
                        size_t vertices_num, dim3 grid_size, dim3 block_size) {
    T *d_small_values;
    size_t values_mem_size = sizeof(T) * vertices_num;
    cuCheck(cudaMalloc(&d_small_values, values_mem_size));

    GetSmallestEdgesLengths<T><<<grid_size, block_size>>>(
        d_cuda_rings, nrings, d_vertices, d_small_values);

    cuCheck(cudaDeviceSynchronize());
    std::vector<T> smalls(vertices_num, 0.0);
    cudaMemcpy(smalls.data(), d_small_values, values_mem_size,
               cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cuCheck(cudaFree(d_small_values));

    return *std::min_element(smalls.begin(), smalls.end());
}

template <typename T>
void MeanCurvatureSmooth<T>::operator()(
    std::shared_ptr<open3d::geometry::TriangleMesh> mesh, int nitr /* = 50 */,
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

    // get device data
    auto [d_cuda_rings, d_edges] = GetCudaRingsAndEdges<T>(mesh);
    auto [d_vertices, d_normals, mem_size] = CopyMeshDataToDevice<T>(mesh);

    // Cuda kernel config
    size_t nvertices = mesh->vertices_.size();
    dim3 block_size(128);
    dim3 grid_size((nvertices + block_size.x - 1) / block_size.x);

    // pre optimization
    T smallest_edge_length = GetSmallestEdgeLength<T>(
        d_cuda_rings, nvertices, d_vertices, nvertices, grid_size, block_size);
    T scalar_fix = scale_factor / smallest_edge_length;
    AddAmount<T><<<grid_size, block_size>>>(d_vertices, nvertices, scalar_fix);

    // optimization
    auto f = [&](auto d_cuda_rings, auto d_vertices, auto d_normals) {
        for (int i = 0; i < nitr; i++) {
            MeanCurvature<T><<<grid_size, block_size>>>(d_cuda_rings, nvertices,
                                                        d_vertices, d_normals);
        }
    };
    TimeCudaEvent(f(d_cuda_rings, d_vertices, d_normals), nitr);

    // post optimization
    UpdateNormals<T><<<grid_size, block_size>>>(d_cuda_rings, nvertices,
                                                d_vertices, d_normals);
    AddAmount<T>
        <<<grid_size.x, block_size>>>(d_vertices, nvertices, 1.0 / scalar_fix);

    // update mesh with optimized data
    cuCheck(cudaDeviceSynchronize());
    mesh->vertices_ = CopyDataToHost(d_vertices, nvertices);
    mesh->vertex_normals_ = CopyDataToHost(d_normals, nvertices);

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

template class MeanCurvatureSmooth<double>;
template class MeanCurvatureSmooth<float>;