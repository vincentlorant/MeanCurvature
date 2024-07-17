#pragma once

#include <oneapi/tbb/concurrent_hash_map.h>
#include <open3d/Open3D.h>

#include <array>
#include <unordered_map>
#include <vector>

#include "types.h"

class HostRing {
 public:
  class EdgeData {
   public:
    void AddAdj(const VertexId &id, const VertexId &pos);
    [[nodiscard]] auto Complete() const -> bool;

    std::array<VertexId, 2> adjs;
    uint8_t count = 0;
  };

 public:
  void AddCenter(VertexId id);

  auto GetCenter() const -> VertexId;

  void AddEdge(VertexId edge_id, VertexId adj, const VertexId &pos);

  auto GetEdges() const -> const std::unordered_map<VertexId, EdgeData> &;

  auto GetCompleteEdgeCount() const -> size_t;

  auto CheckForIncompleteEdges() const -> std::vector<int>;

  std::unordered_map<VertexId, EdgeData> edges_;
  VertexId center_ = -1;
};

using HostRingsMap = tbb::concurrent_hash_map<int, HostRing>;

void AddEdges(HostRingsMap &ids_neighbors,
  HostRingsMap::accessor &ac,
  const VertexId &a,
  const VertexId &b,
  const VertexId &c);

auto CreateHostRings(
  std::shared_ptr<open3d::geometry::TriangleMesh> mesh) -> HostRingsMap;