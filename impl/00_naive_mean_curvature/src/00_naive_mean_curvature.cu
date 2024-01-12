#include <Eigen/Dense>
#include <memory>

#include <oneapi/tbb.h>
#include <open3d/Open3D.h>

#include "cuda_helpers.h"
#include "cuda_ring.h"
#include "helpers.hpp"
#include "host_ring.h"
#include "mean_curvature_types.h"

__global__ void OptimizeMeanCurvature(CudaRing *rings, int n,
                                      Eigen::Vector3d *positions,
                                      Eigen::Vector3d *normals) {
    int pos = threadIdx.x + blockIdx.x * blockDim.x;
    if (pos < n) {
        rings[pos].ComputeCurvature(positions, normals);
    }
}

__global__ void GetSmallestEdgesLengths(CudaRing *rings, int n,
                                        Eigen::Vector3d *positions,
                                        double *out) {
    int pos = threadIdx.x + blockIdx.x * blockDim.x;
    if (pos < n) {
        out[pos] = rings[pos].GetSmallestEdgeLength(positions);
    }
}

__global__ void ScalePositions(Eigen::Vector3d *positions, int n,
                               float amount) {
    int pos = threadIdx.x + blockIdx.x * blockDim.x;
    if (pos < n) {
        positions[pos] *= amount;
    }
}

double GetSmallestEdgeLength(CudaRing *d_cuda_rings, size_t nrings,
                             Eigen::Vector3d *d_vertices, size_t nvertices,
                             dim3 grid_size, dim3 block_size) {
    double *d_small_values;
    size_t values_mem_size = sizeof(double) * nvertices;
    cudaCheck(cudaMalloc(&d_small_values, values_mem_size));
    GetSmallestEdgesLengths<<<grid_size, block_size>>>(
        d_cuda_rings, nrings, d_vertices, d_small_values);

    cudaCheck(cudaDeviceSynchronize());
    std::vector<double> smalls(nvertices, 0);
    cudaMemcpy(smalls.data(), d_small_values, values_mem_size,
               cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaCheck(cudaFree(d_small_values));

    return *std::min_element(smalls.begin(), smalls.end());
}

std::vector<CudaRing>
GetCudaRings(std::shared_ptr<open3d::geometry::TriangleMesh> mesh) {
    auto host_rings = CreateHostRings(mesh);
    size_t nrings = host_rings.size();
    std::vector<CudaRing> cuda_rings;
    cuda_rings.reserve(nrings);
    for (auto &r : host_rings) {
        cuda_rings.push_back(r.second.ToCudaRing());
    }

    return cuda_rings;
}

void MeanCurvature(std::shared_ptr<open3d::geometry::TriangleMesh> mesh,
                   int itr = 50, double scalar = 0.035) {
    std::vector<CudaRing> cuda_rings = GetCudaRings(mesh);
    size_t nvertices = mesh->vertices_.size();
    size_t nrings = cuda_rings.size();

    // copy ring, vertices & normals data to the device
    CudaRing *d_cuda_rings;
    Eigen::Vector3d *d_vertices;
    Eigen::Vector3d *d_normals;
    size_t ring_mem_size = sizeof(CudaRing) * nrings;
    cudaCheck(cudaMalloc(&d_cuda_rings, ring_mem_size));
    cudaCheck(cudaMemcpy(d_cuda_rings, cuda_rings.data(), ring_mem_size,
                         cudaMemcpyKind::cudaMemcpyHostToDevice));
    size_t vec_mem_size = sizeof(Eigen::Vector3d) * nvertices;
    cudaCheck(cudaMalloc(&d_vertices, vec_mem_size));
    cudaCheck(cudaMalloc(&d_normals, vec_mem_size));
    cudaCheck(cudaMemcpy(d_vertices, mesh->vertices_.data(), vec_mem_size,
                         cudaMemcpyKind::cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_normals, mesh->vertex_normals_.data(), vec_mem_size,
                         cudaMemcpyKind::cudaMemcpyHostToDevice));

    // cuda kernel config
    dim3 block_size(128);
    dim3 grid_size((nrings + block_size.x - 1) / block_size.x);

    /* Note
     * depending on input data unit and scaling, edges length can be very small
     * to avoid problems which could occurs during computation, the smallest
     * edge's length is used to scale up the vertices
     */
    double smallest = GetSmallestEdgeLength(d_cuda_rings, nrings, d_vertices,
                                            nvertices, grid_size, block_size);
    double scalar_fix = scalar / smallest;
    ScalePositions<<<grid_size, block_size>>>(d_vertices, nrings, scalar_fix);

    // optimization
    for (int i = 0; i < itr; i++) {
        OptimizeMeanCurvature<<<grid_size, block_size>>>(d_cuda_rings, nrings,
                                                         d_vertices, d_normals);
    }
    cudaCheck(cudaDeviceSynchronize());

    // scale back vertices to original values
    ScalePositions<<<grid_size, block_size>>>(d_vertices, nrings,
                                              1.0 / scalar_fix);
    cudaCheck(cudaDeviceSynchronize());

    // get back updated vertices to host
    cudaCheck(cudaMemcpy(mesh->vertices_.data(), d_vertices, vec_mem_size,
                         cudaMemcpyKind::cudaMemcpyDeviceToHost));
    cudaCheck(cudaFree(d_cuda_rings));
    cudaCheck(cudaFree(d_vertices));
    cudaCheck(cudaFree(d_normals));

    // update normals with new vertices
    mesh->ComputeVertexNormals();
}

int main(int argv, char **argc) {
    SetupLogger();
    int itr = argv > 1 ? std::stoi(argc[1]) : 50;
    auto filename = "../../assets/se_sub.ply";
    auto mesh = GetMesh(filename, 0.01);
    TimeIt(MeanCurvature(mesh, itr));
    open3d::visualization::DrawGeometries({mesh}, "debug", 1280, 720, 100, 100,
                                          true);

    return EXIT_SUCCESS;
}