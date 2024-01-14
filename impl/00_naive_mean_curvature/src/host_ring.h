#pragma once

#include <unordered_map>
#include <vector>

#include <glog/logging.h>
#include <oneapi/tbb/concurrent_hash_map.h>
#include <open3d/Open3D.h>

#include "cuda_ring.h"
#include "mean_curvature_types.h"

class HostRing {
    struct EdgeData {
        VertexId adjs[2];
        uint8_t count = 0;

        void AddAdj(const VertexId &id);
        inline bool Complete() const;
    };

    std::unordered_map<VertexId, EdgeData> edges_;
    VertexId center_;

  public:
    void AddCenter(VertexId id);

    inline VertexId GetCenter() const;

    const std::unordered_map<VertexId, EdgeData> &GetEdges() const;

    void AddEdge(VertexId edge_id, VertexId adj);

    CudaRing ToCudaRing() const;

    // happen when mesh is not closed
    std::vector<int> CheckForIncompleteEdges() const;
};

using HostRingsMap = tbb::concurrent_hash_map<int, HostRing>;

void AddEdges(HostRingsMap &ids_neighbors, HostRingsMap::accessor &ac,
              const VertexId &a, const VertexId &b, const VertexId &c);

HostRingsMap
CreateHostRings(std::shared_ptr<open3d::geometry::TriangleMesh> mesh);