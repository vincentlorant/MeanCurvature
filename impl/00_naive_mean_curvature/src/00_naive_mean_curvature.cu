#include <Eigen/Dense>
#include <memory>
#include <tuple>

#include <oneapi/tbb.h>
#include <open3d/Open3D.h>

#include "cuda_helpers.h"
#include "cuda_ring.h"
#include "helpers.hpp"
#include "host_ring.h"
#include "types.h"

__global__ void OptimizeMeanCurvature(CudaRing *rings, int n,
                                      Eigen::Vector3d *positions,
                                      Eigen::Vector3d *normals) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        rings[i].ComputeCurvature(positions, normals);
    }
}

__global__ void GetSmallestEdgesLengths(CudaRing *rings, int n,
                                        Eigen::Vector3d *positions,
                                        double *out) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        out[i] = rings[i].GetSmallestEdgeLength(positions);
    }
}

__global__ void ScalePositions(Eigen::Vector3d *positions, int n,
                               float amount) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        positions[i] *= amount;
    }
}

double GetSmallestEdgeLength(CudaRing *d_cuda_rings, size_t nrings,
                             Eigen::Vector3d *d_vertices, size_t vertices_num,
                             dim3 grid_size, dim3 block_size) {
    double *d_small_values;
    size_t values_mem_size = sizeof(double) * vertices_num;
    cuCheck(cudaMalloc(&d_small_values, values_mem_size));
    GetSmallestEdgesLengths<<<grid_size, block_size>>>(
        d_cuda_rings, nrings, d_vertices, d_small_values);

    cuCheck(cudaDeviceSynchronize());
    std::vector<double> smalls(vertices_num, 0);
    cudaMemcpy(smalls.data(), d_small_values, values_mem_size,
               cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cuCheck(cudaFree(d_small_values));

    return *std::min_element(smalls.begin(), smalls.end());
}

CudaRing ToCudaRing(const HostRing &ring, CudaEdge *d_edges) {

    return CudaRing(ring.GetCenter(), d_edges, ring.GetCompleteEdgeCount());
}

std::pair<CudaRing *, CudaEdge *>
GetCudaRingsAndEdges(std::shared_ptr<open3d::geometry::TriangleMesh> mesh) {
    auto host_rings = CreateHostRings(mesh);
    size_t nrings = host_rings.size();
    std::vector<CudaRing> cuda_rings;
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
    cuCheck(cudaMalloc(&d_edges, mem_size));
    cuCheck(cudaMemcpy(d_edges, edges.get(), mem_size,
                       cudaMemcpyKind::cudaMemcpyHostToDevice));

    // create cuda rings
    CudaEdge *pos = d_edges;
    for (const auto &r : host_rings) {
        cuda_rings.push_back(ToCudaRing(r.second, pos));
        pos += r.second.GetCompleteEdgeCount();
    }
    CudaRing *d_cuda_rings;
    size_t ring_mem_size = (sizeof(CudaRing) * nrings);
    cuCheck(cudaMalloc(&d_cuda_rings, ring_mem_size));
    cuCheck(cudaMemcpy(d_cuda_rings, cuda_rings.data(), ring_mem_size,
                       cudaMemcpyKind::cudaMemcpyHostToDevice));

    return {d_cuda_rings, d_edges};
}

std::tuple<Eigen::Vector3d *, Eigen::Vector3d *, size_t>
CopyMeshDataToDevice(std::shared_ptr<open3d::geometry::TriangleMesh> mesh) {
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

void MeanCurvatureSmoothing(
    std::shared_ptr<open3d::geometry::TriangleMesh> mesh, int itr = 50,
    double scalar = 0.035) {
    auto [d_cuda_rings, d_edges] = GetCudaRingsAndEdges(mesh);
    auto [d_vertices, d_normals, mem_size] = CopyMeshDataToDevice(mesh);

    // Cuda kernel config
    size_t vertices_num = mesh->vertices_.size();
    dim3 block_size(256);
    dim3 grid_size((vertices_num + block_size.x - 1) / block_size.x);

    /* Purpose:
     * Uniformly scales up vertices based on the smallest edge's length to
     * ensure computation-friendly values, especially when dealing with small
     * edge lengths.
     */
    double smallest =
        GetSmallestEdgeLength(d_cuda_rings, vertices_num, d_vertices,
                              vertices_num, grid_size, block_size);
    double scalar_fix = scalar / smallest;
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
    cuCheck(cudaDeviceSynchronize());

    // Get back updated vertices to host
    cuCheck(cudaMemcpy(mesh->vertices_.data(), d_vertices, mem_size,
                       cudaMemcpyKind::cudaMemcpyDeviceToHost));
    cuCheck(cudaFree(d_cuda_rings));
    cuCheck(cudaFree(d_edges));
    cuCheck(cudaFree(d_vertices));
    cuCheck(cudaFree(d_normals));

    // Update normals with new vertices
    mesh->ComputeVertexNormals();
}

int main(int argv, char **argc) {
    SetupLogger();
    int itr = argv > 1 ? std::stoi(argc[1]) : 50;
    auto filename = "../../assets/se_sub.ply";
    auto mesh = GetMesh(filename, 0.01, true);
    LOG(INFO) << mesh->vertices_.size();
    TimeIt(MeanCurvatureSmoothing(mesh, itr));
    open3d::visualization::DrawGeometries({mesh}, "debug", 1280, 720, 100, 100,
                                          true);

    return EXIT_SUCCESS;
}