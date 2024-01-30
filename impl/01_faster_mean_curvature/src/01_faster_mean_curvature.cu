#include <Eigen/Dense>
#include <memory>
#include <tuple>
#include <type_traits>

#include <oneapi/tbb.h>
#include <open3d/Open3D.h>

#include "cuda_helpers.h"
#include "cuda_ring.h"
#include "helpers.hpp"
#include "host_ring.h"

template <typename T>
__global__ void MeanCurvatureKernel(CudaRing<T> *rings, int n,
                                    typename CudaRing<T>::Vector *positions,
                                    typename CudaRing<T>::Vector *normals) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        rings[i].ComputeCurvature(positions, normals);
    }
}

template <typename T>
__global__ void AddAmountKernel(typename CudaRing<T>::Vector *arr, int n,
                                float amount) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        arr[i] *= amount;
    }
}

/*
 * When mean-curvature is processed with double floating point precision
 * we can directly copy vertices and normal to device allocated arrays
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
 * When mean-curvature is processed with single floating points precision
 * we have to convert vertices and normal before to copy them to device
 * allocated arrays
 */
template <typename T = float, typename = typename std::enable_if<
                                  std::is_same<T, float>::value>::type>
std::tuple<Eigen::Vector3f *, Eigen::Vector3f *, size_t>
CopyMeshDataToDevice(std::shared_ptr<open3d::geometry::TriangleMesh> mesh) {
    size_t n = mesh->vertices_.size();
    Eigen::Map<Eigen::Matrix3Xd> vi(mesh->vertices_[0].data(), 3, n);
    Eigen::Map<Eigen::Matrix3Xd> ni(mesh->vertex_normals_[0].data(), 3, n);
    std::vector<Eigen::Vector3f> vf(vi.cast<float>().colwise().begin(),
                                    vi.cast<float>().colwise().end());
    std::vector<Eigen::Vector3f> nf(ni.cast<float>().colwise().begin(),
                                    ni.cast<float>().colwise().end());

    Eigen::Vector3f *d_vertices;
    Eigen::Vector3f *d_normals;
    size_t vec_mem_size = sizeof(Eigen::Vector3f) * n;
    cuCheck(cudaMalloc(&d_vertices, vec_mem_size));
    cuCheck(cudaMalloc(&d_normals, vec_mem_size));
    cuCheck(cudaMemcpy(d_vertices, vf.data(), vec_mem_size,
                       cudaMemcpyKind::cudaMemcpyHostToDevice));
    cuCheck(cudaMemcpy(d_normals, nf.data(), vec_mem_size,
                       cudaMemcpyKind::cudaMemcpyHostToDevice));

    return {d_vertices, d_normals, vec_mem_size};
}

// define cuda rings and edges and copy them to device
template <typename T>
std::pair<CudaRing<T> *, CudaEdge *>
GetCudaRingsAndEdges(std::shared_ptr<open3d::geometry::TriangleMesh> mesh) {
    auto host_rings = CreateHostRings(mesh);
    size_t nrings = host_rings.size();
    std::vector<CudaRing<T>> cuda_rings;
    cuda_rings.reserve(nrings);
    // create cuda edges
    size_t edges_count = 0;
    for (auto &r : host_rings) {
        edges_count += r.second.GetCompleteEdgeCount();
    }
    auto edges = std::vector<CudaEdge>(edges_count);
    unsigned c = 0;
    for (const auto &r : host_rings) {
        for (const auto &e : r.second.GetEdges()) {
            if (e.second.Complete()) {
                edges[c++] = {e.first, e.second.adjs[0], e.second.adjs[1]};
            }
        }
    }
    assert(c == edges_count);
    CudaEdge *d_edges;
    size_t mem_size = sizeof(CudaEdge) * edges_count;
    cuCheck(cudaMalloc(&d_edges, mem_size));
    cuCheck(cudaMemcpy(d_edges, edges.data(), mem_size,
                       cudaMemcpyKind::cudaMemcpyHostToDevice));
    // create cuda rings
    CudaEdge *pos = d_edges;
    for (const auto &r : host_rings) {
        cuda_rings.push_back(ToCudaRing<T>(r.second, pos));
        pos += r.second.GetCompleteEdgeCount();
    }
    CudaRing<T> *d_cuda_rings;
    size_t ring_mem_size = (sizeof(CudaRing<T>) * nrings);
    cuCheck(cudaMalloc(&d_cuda_rings, ring_mem_size));
    cuCheck(cudaMemcpy(d_cuda_rings, cuda_rings.data(), ring_mem_size,
                       cudaMemcpyKind::cudaMemcpyHostToDevice));

    return {d_cuda_rings, d_edges};
}

void UpdateMeshVertices(std::shared_ptr<open3d::geometry::TriangleMesh> mesh,
                        const std::vector<Eigen::Vector3d> &vertices) {
    mesh->vertices_ = vertices;
}

void UpdateMeshVertices(std::shared_ptr<open3d::geometry::TriangleMesh> mesh,
                        std::vector<Eigen::Vector3f> &vertices) {
    Eigen::Map<Eigen::Matrix3Xf> vi(vertices[0].data(), 3, vertices.size());

    mesh->vertices_ = std::vector<Eigen::Vector3d>(
        vi.cast<double>().colwise().begin(), vi.cast<double>().colwise().end());
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
    std::vector<T> smalls(vertices_num, 0);
    cudaMemcpy(smalls.data(), d_small_values, values_mem_size,
               cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cuCheck(cudaFree(d_small_values));

    return *std::min_element(smalls.begin(), smalls.end());
}

template <typename T>
void MeanCurvatureSmoothing(
    std::shared_ptr<open3d::geometry::TriangleMesh> mesh, int itr = 50,
    float scale_factor = 0.035) {
    auto [d_cuda_rings, d_edges] = GetCudaRingsAndEdges<T>(mesh);
    auto [d_vertices, d_normals, mem_size] = CopyMeshDataToDevice<T>(mesh);
    // Cuda kernel config
    size_t vertices_num = mesh->vertices_.size();
    dim3 block_size(128);
    dim3 grid_size((vertices_num + block_size.x - 1) / block_size.x);

    // scale mesh vertices position to a computation friendly size
    T smallest_edge_length =
        GetSmallestEdgeLength<T>(d_cuda_rings, vertices_num, d_vertices,
                                 vertices_num, grid_size, block_size);
    T scalar_fix = scale_factor / smallest_edge_length;
    AddAmountKernel<T>
        <<<grid_size.x, block_size>>>(d_vertices, vertices_num, scalar_fix);

    // Optimize
    auto f = [&](auto d_cuda_rings, auto d_vertices, auto d_normals) {
        for (int i = 0; i < itr; i++) {
            MeanCurvatureKernel<T><<<grid_size, block_size>>>(
                d_cuda_rings, vertices_num, d_vertices, d_normals);
        }
    };
    TimeCudaEvent(f(d_cuda_rings, d_vertices, d_normals), itr);

    // Scale back vertices to original values
    AddAmountKernel<T><<<grid_size.x, block_size>>>(d_vertices, vertices_num,
                                                    1.0 / scalar_fix);
    cuCheck(cudaDeviceSynchronize());

    // Update mesh vertices with device data
    cuCheck(cudaMemcpy(mesh->vertices_.data(), d_vertices, mem_size,
                       cudaMemcpyKind::cudaMemcpyDeviceToHost));
    std::vector<typename CudaRing<T>::Vector> vertices(vertices_num);
    cuCheck(cudaMemcpy(vertices.data(), d_vertices, mem_size,
                       cudaMemcpyKind::cudaMemcpyDeviceToHost));
    UpdateMeshVertices(mesh, vertices);

    // clean allocated device data
    cuCheck(cudaFree(d_cuda_rings));
    cuCheck(cudaFree(d_edges));
    cuCheck(cudaFree(d_vertices));
    cuCheck(cudaFree(d_normals));

    // Update normals using new vertices position
    mesh->ComputeVertexNormals();
}

int main(int argv, char **argc) {
    SetupLogger();
    // inputs
    int itr = argv > 1 ? std::stoi(argc[1]) : 50;
    float scale_factor = argv > 2 ? std::stof(argc[2]) : 0.035;
    auto filename = "../../assets/se_sub.ply";
    // run algorithm
    auto mesh = GetMesh(filename, 0.01, false);
    TimeIt(MeanCurvatureSmoothing<float>(mesh, itr, scale_factor));
    // vizualize
    open3d::visualization::DrawGeometries({mesh}, "debug", 1280, 720, 100, 100,
                                          true);

    return EXIT_SUCCESS;
}