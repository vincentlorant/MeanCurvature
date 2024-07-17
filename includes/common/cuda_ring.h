#include <cuda_runtime.h>
#include <open3d/Open3D.h>

#include "common/host_ring.h"
#include "common/types.h"

auto ToCudaRing(const HostRing &ring, CudaEdge *d_edges) -> CudaRing;

auto CreateCudaEdges(const HostRingsMap &host_rings) -> CudaEdge *;

auto CreateCudaRings(
  const HostRingsMap &host_rings, CudaEdge *d_edges) -> CudaRing *;

auto GetCudaRingsAndEdges(std::shared_ptr<open3d::geometry::TriangleMesh> mesh)
  -> std::pair<CudaRing *, CudaEdge *>;