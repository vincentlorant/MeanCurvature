#pragma once

#include <unordered_map>
#include <vector>

#include <glog/logging.h>
#include <oneapi/tbb/concurrent_hash_map.h>
#include <open3d/Open3D.h>

#include "types.h"

class HostRing {
  public:
    struct EdgeData {
        VertexId adjs[2];
        uint8_t count = 0;

        void AddAdj(const VertexId &id);
        void AddAdj(const VertexId &id, const VertexId &pos);
        bool Complete() const;
    };

  public:
    void AddCenter(VertexId id);

    VertexId GetCenter() const;

    void AddEdge(VertexId edge_id, VertexId adj);

    void AddEdge(VertexId edge_id, VertexId adj, const VertexId &pos);

    const std::unordered_map<VertexId, EdgeData> &GetEdges() const;

    size_t GetCompleteEdgeCount() const;

    // happens when input meshes are not closed
    std::vector<int> CheckForIncompleteEdges() const;

  private:
    std::unordered_map<VertexId, EdgeData> edges_;
    VertexId center_ = -1;
};

using HostRingsMap = tbb::concurrent_hash_map<int, HostRing>;

void AddEdges(HostRingsMap &ids_neighbors, HostRingsMap::accessor &ac,
              const VertexId &a, const VertexId &b, const VertexId &c);

HostRingsMap
CreateHostRings(std::shared_ptr<open3d::geometry::TriangleMesh> mesh);