#pragma once

#include "mean_curvature_types.h"
#include "oneapi/tbb/concurrent_hash_map.h"
#include <glog/logging.h>

struct EdgeData {
    VertexId adjs[2];
    uint8_t count = 0;

    void AddAdj(const VertexId &id) {
        uint8_t array_num = sizeof(adjs) / sizeof(adjs[0]);
        if (count < array_num) {
            adjs[count++] = id;
        }
    }

    inline bool Complete() const { return count == 2; }
};

class HostRing {
    std::unordered_map<VertexId, EdgeData> edges_;
    VertexId center_;

  public:
    void AddCenter(VertexId id) { center_ = id; }

    inline VertexId GetCenter() const { return center_; }

    const std::unordered_map<VertexId, EdgeData> &GetEdges() const {
        return edges_;
    }

    void AddEdge(VertexId edge_id, VertexId adj) {
        if (edges_.find(edge_id) == edges_.end()) {
            edges_[edge_id] = EdgeData();
        }
        edges_[edge_id].AddAdj(adj);
    }

    CudaRing ToCudaRing() const {
        if (edges_.size() > NUM_EDGES) {
            LOG(FATAL) << "edges size:" << edges_.size()
                       << " NUM_EDGES def:" << NUM_EDGES;
        }
        CudaRing cr(center_);
        for (const auto &p : edges_) {
            if (p.second.Complete() == false) {
                continue;
            }
            cr.AddEdge(p.first, p.second.adjs[0], p.second.adjs[1]);
        }

        return cr;
    }

    // happen when mesh is not closed
    std::vector<int> CheckForIncompleteEdges() const {
        std::vector<int> incompletes;
        for (const auto &p : edges_) {
            if (p.second.Complete()) {
                incompletes.push_back(p.first);
            }
        }

        return incompletes;
    }
};

using HostRingsMap = tbb::concurrent_hash_map<int, HostRing>;

void AddEdges(HostRingsMap &ids_neighbors, HostRingsMap::accessor &ac,
              const VertexId &a, const VertexId &b, const VertexId &c) {
    if (!ids_neighbors.find(ac, a)) {
        ids_neighbors.insert(ac, a);
        ac->second.AddCenter(a);
    }
    ac->second.AddEdge(b, c);
    ac->second.AddEdge(c, b);
}

HostRingsMap
CreateHostRings(std::shared_ptr<open3d::geometry::TriangleMesh> mesh) {
    const auto &triangles = mesh->triangles_;
    HostRingsMap ids_neighbors;

    auto func = [&](size_t i) {
        auto &t = triangles[i];
        HostRingsMap::accessor ac;
        AddEdges(ids_neighbors, ac, t[0], t[1], t[2]);
        AddEdges(ids_neighbors, ac, t[1], t[0], t[2]);
        AddEdges(ids_neighbors, ac, t[2], t[0], t[1]);
    };

    tbb::parallel_for(size_t(0), triangles.size(), func);

    return ids_neighbors;
}