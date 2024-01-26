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
#include "mean_curvature_types.h"

using ComputeType = float;
using CudaRingT = CudaRing<ComputeType>;
#define IS_DOUBLE(type) std::is_same<type, double>::value

__global__ void OptimizeMeanCurvature(CudaRingT *rings, int n,
                                      CudaRingT::Vector *positions,
                                      CudaRingT::Vector *normals) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        rings[i].ComputeCurvature(positions, normals);
    }
}

__global__ void GetSmallestEdgesLengths(CudaRingT *rings, int n,
                                        CudaRingT::Vector *positions,
                                        ComputeType *out) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        out[i] = rings[i].GetSmallestEdgeLength(positions);
    }
}

__global__ void ScalePositions(CudaRingT::Vector *positions, int n,
                               float amount) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        positions[i] *= amount;
    }
}

ComputeType GetSmallestEdgeLength(CudaRingT *d_cuda_rings, size_t nrings,
                                  CudaRingT::Vector *d_vertices,
                                  size_t vertices_num, dim3 grid_size,
                                  dim3 block_size) {
    ComputeType *d_small_values;
    size_t values_mem_size = sizeof(ComputeType) * vertices_num;
    cudaCheck(cudaMalloc(&d_small_values, values_mem_size));
    GetSmallestEdgesLengths<<<grid_size, block_size>>>(
        d_cuda_rings, nrings, d_vertices, d_small_values);

    cudaCheck(cudaDeviceSynchronize());
    std::vector<ComputeType> smalls(vertices_num, 0);
    cudaMemcpy(smalls.data(), d_small_values, values_mem_size,
               cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaCheck(cudaFree(d_small_values));

    return *std::min_element(smalls.begin(), smalls.end());
}

std::pair<CudaRingT *, CudaEdge *>
GetCudaRingsAndEdges(std::shared_ptr<open3d::geometry::TriangleMesh> mesh) {
    auto host_rings = CreateHostRings(mesh);
    size_t nrings = host_rings.size();
    std::vector<CudaRingT> cuda_rings;
    cuda_rings.reserve(nrings);

    // create cuda edges
    size_t edges_count = 0;
    for (auto &r : host_rings) {
        edges_count += r.second.GetCompleteEdgeCount();
    }
    auto edges = std::make_unique<CudaEdge[]>(edges_count);
    int c = 0;
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
    cudaCheck(cudaMalloc(&d_edges, mem_size));
    cudaCheck(cudaMemcpy(d_edges, edges.get(), mem_size,
                         cudaMemcpyKind::cudaMemcpyHostToDevice));

    // create cuda rings
    CudaEdge *pos = d_edges;
    for (const auto &r : host_rings) {
        cuda_rings.push_back(ToCudaRing<ComputeType>(r.second, pos));
        pos += r.second.GetCompleteEdgeCount();
    }
    CudaRingT *d_cuda_rings;
    size_t ring_mem_size = (sizeof(CudaRingT) * nrings);
    cudaCheck(cudaMalloc(&d_cuda_rings, ring_mem_size));
    cudaCheck(cudaMemcpy(d_cuda_rings, cuda_rings.data(), ring_mem_size,
                         cudaMemcpyKind::cudaMemcpyHostToDevice));

    return {d_cuda_rings, d_edges};
}

std::tuple<CudaRingT::Vector *, CudaRingT::Vector *, size_t>
CopyMeshDataToDevice(std::shared_ptr<open3d::geometry::TriangleMesh> mesh) {
    size_t n = mesh->vertices_.size();
    CudaRingT::Vector *d_vertices;
    CudaRingT::Vector *d_normals;

    // TODO
    CudaRingT::Vector *vertices;
    CudaRingT::Vector *normals;

    bool process_as_double = std::is_same<ComputeType, double>::value;
#if process_as_double
    vertices = mesh->vertices_.data();
    normals = mesh->vertex_normals_.data();
#else
    Eigen::Map<Eigen::Matrix3Xd> vi(mesh->vertices_[0].data(), 3, n);
    Eigen::Map<Eigen::Matrix3Xd> ni(mesh->vertex_normals_[0].data(), 3, n);
    std::vector<Eigen::Vector3f> vf(vi.cast<float>().colwise().begin(),
                                    vi.cast<float>().colwise().end());
    std::vector<Eigen::Vector3f> nf(ni.cast<float>().colwise().begin(),
                                    ni.cast<float>().colwise().end());
    vertices = vf.data();
    normals = nf.data();
#endif

    size_t vec_mem_size = sizeof(CudaRingT::Vector) * n;
    cudaCheck(cudaMalloc(&d_vertices, vec_mem_size));
    cudaCheck(cudaMalloc(&d_normals, vec_mem_size));
    cudaCheck(cudaMemcpy(d_vertices, vertices, vec_mem_size,
                         cudaMemcpyKind::cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_normals, normals, vec_mem_size,
                         cudaMemcpyKind::cudaMemcpyHostToDevice));

    return {d_vertices, d_normals, vec_mem_size};
}

void MeanCurvature(std::shared_ptr<open3d::geometry::TriangleMesh> mesh,
                   int itr = 50, double scalar = 0.035) {
    auto [d_cuda_rings, d_edges] = GetCudaRingsAndEdges(mesh);
    auto [d_vertices, d_normals, mem_size] = CopyMeshDataToDevice(mesh);

    // Cuda kernel config
    size_t vertices_num = mesh->vertices_.size();
    dim3 block_size(128);
    dim3 grid_size((vertices_num + block_size.x - 1) / block_size.x);

    /* Purpose:
     * Uniformly scales up vertices based on the smallest edge's length to
     * ensure computation-friendly values, especially when dealing with small
     * edge lengths.
     */
    ComputeType smallest =
        GetSmallestEdgeLength(d_cuda_rings, vertices_num, d_vertices,
                              vertices_num, grid_size, block_size);
    ComputeType scalar_fix = scalar / smallest;
    ScalePositions<<<grid_size, block_size>>>(d_vertices, vertices_num,
                                              scalar_fix);

    // Optimization
    auto f = [&](auto d_cuda_rings, auto d_vertices, auto d_normals) {
        for (int i = 0; i < itr; i++) {
            OptimizeMeanCurvature<<<grid_size, block_size>>>(
                d_cuda_rings, vertices_num, d_vertices, d_normals);
        }
    };
    TimeCudaEvent(f(d_cuda_rings, d_vertices, d_normals), itr);

    // Scale back vertices to original values
    ScalePositions<<<grid_size, block_size>>>(d_vertices, vertices_num,
                                              1.0 / scalar_fix);
    cudaCheck(cudaDeviceSynchronize());

    // Get back updated vertices to host
    // TODO
    cudaCheck(cudaMemcpy(mesh->vertices_.data(), d_vertices, mem_size,
                         cudaMemcpyKind::cudaMemcpyDeviceToHost));

    std::vector<CudaRingT::Vector > vertices(vertices_num);

    cudaCheck(cudaMemcpy(vertices.data(), d_vertices, mem_size,
                         cudaMemcpyKind::cudaMemcpyDeviceToHost));

    bool process_as_double = std::is_same<ComputeType, double>::value;
    #if process_as_double
        mesh->vertices_ = vertices;
    #else
        Eigen::Map<Eigen::Matrix3Xf> vi(vertices[0].data(), 3, vertices_num);
        std::vector<Eigen::Vector3d> vd(vi.cast<double>().colwise().begin(),
                                        vi.cast<double>().colwise().end());
       mesh->vertices_ = vd;
    #endif

    cudaCheck(cudaFree(d_cuda_rings));
    cudaCheck(cudaFree(d_edges));
    cudaCheck(cudaFree(d_vertices));
    cudaCheck(cudaFree(d_normals));

    // Update normals using new vertices position
    mesh->ComputeVertexNormals();
}

int main(int argv, char **argc) {
    SetupLogger();
    int itr = argv > 1 ? std::stoi(argc[1]) : 50;
    float scalar = argv > 2 ? std::stof(argc[2]) : 0.035;
    auto filename = "../../assets/se_sub.ply";
    auto mesh = GetMesh(filename, 0.01, true);
    TimeIt(MeanCurvature(mesh, itr, scalar));
    open3d::visualization::DrawGeometries({mesh}, "debug", 1280, 720, 100, 100,
                                          true);

    return EXIT_SUCCESS;
}