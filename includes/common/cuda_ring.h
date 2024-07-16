#include <cuda_runtime.h>
#include <open3d/Open3D.h>

#include "common/host_ring.h"
#include "common/types.h"

CudaRing ToCudaRing(const HostRing &ring, CudaEdge *d_edges);

CudaEdge *CreateCudaEdges(const HostRingsMap &host_rings);

CudaRing *CreateCudaRings(const HostRingsMap &host_rings, CudaEdge *d_edges);

std::pair<CudaRing *, CudaEdge *> GetCudaRingsAndEdges(
    std::shared_ptr<open3d::geometry::TriangleMesh> mesh);