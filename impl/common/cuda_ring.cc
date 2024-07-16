#include <cuda_runtime.h>
#include <open3d/Open3D.h>

#include "common/cuda/cuda_helpers.h"
#include "common/host_ring.h"
#include "common/types.h"

CudaRing ToCudaRing(const HostRing &ring, CudaEdge *d_edges) {
  return {ring.GetCenter(), d_edges, ring.GetCompleteEdgeCount()};
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

CudaRing *CreateCudaRings(const HostRingsMap &host_rings, CudaEdge *d_edges) {
  CudaEdge *pos = d_edges;
  size_t nrings = host_rings.size();
  std::vector<CudaRing> cuda_rings;
  cuda_rings.reserve(nrings);
  for (const auto &r : host_rings) {
    cuda_rings.push_back(ToCudaRing(r.second, pos));
    pos += r.second.GetCompleteEdgeCount();
  }
  CudaRing *d_cuda_rings;
  size_t ring_mem_size = (sizeof(CudaRing) * nrings);
  cuCheck(cudaMalloc(&d_cuda_rings, ring_mem_size));
  cuCheck(cudaMemcpy(d_cuda_rings, cuda_rings.data(), ring_mem_size,
                     cudaMemcpyKind::cudaMemcpyHostToDevice));

  return d_cuda_rings;
}

std::pair<CudaRing *, CudaEdge *> GetCudaRingsAndEdges(
    std::shared_ptr<open3d::geometry::TriangleMesh> mesh) {
  auto host_rings = CreateHostRings(mesh);
  CudaEdge *d_edges = CreateCudaEdges(host_rings);
  CudaRing *d_cuda_rings = CreateCudaRings(host_rings, d_edges);

  return {d_cuda_rings, d_edges};
}