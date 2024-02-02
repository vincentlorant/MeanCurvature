#include "host_ring.h"

#include <oneapi/tbb.h>
#include <open3d/Open3D.h>
#include <unordered_map>
#include <vector>

#include "cuda_helpers.h"

void HostRing::EdgeData::AddAdj(const VertexId &id) {
    if (count < 2) {
        adjs[count++] = id;
    }
}

void HostRing::EdgeData::AddAdj(const VertexId &id, const VertexId &pos) {
    if (pos < 2) {
        adjs[pos] = id;
        count++;
    }
}

bool HostRing::EdgeData::Complete() const { return count == 2; }

void HostRing::AddCenter(VertexId id) { center_ = id; }

VertexId HostRing::GetCenter() const { return center_; }

void HostRing::AddEdge(VertexId edge_id, VertexId adj) {
    if (edges_.find(edge_id) == edges_.end()) {
        edges_[edge_id] = EdgeData();
    }
    edges_[edge_id].AddAdj(adj);
}

void HostRing::AddEdge(VertexId edge_id, VertexId adj, const VertexId &pos) {
    if (edges_.find(edge_id) == edges_.end()) {
        edges_[edge_id] = EdgeData();
    }
    edges_[edge_id].AddAdj(adj, pos);
}

const std::unordered_map<VertexId, HostRing::EdgeData> &
HostRing::GetEdges() const {
    return edges_;
}

size_t HostRing::GetCompleteEdgeCount() const {
    size_t count = 0;
    for (const auto &p : edges_) {
        if (p.second.Complete()) {
            count++;
        }
    }

    return count;
}

std::vector<int> HostRing::CheckForIncompleteEdges() const {
    std::vector<int> incompletes;
    for (const auto &p : edges_) {
        if (p.second.Complete()) {
            incompletes.push_back(p.first);
        }
    }

    return incompletes;
}

void AddEdges(HostRingsMap &ids_neighbors, HostRingsMap::accessor &ac,
              const VertexId &a, const VertexId &b, const VertexId &c) {
    if (!ids_neighbors.find(ac, a)) {
        ids_neighbors.insert(ac, a);
        ac->second.AddCenter(a);
    }
    ac->second.AddEdge(b, c, 1);
    ac->second.AddEdge(c, b, 0);
}

HostRingsMap
CreateHostRings(std::shared_ptr<open3d::geometry::TriangleMesh> mesh) {
    const auto &triangles = mesh->triangles_;
    HostRingsMap ids_neighbors;

    auto func = [&](size_t i) {
        auto &t = triangles[i];
        HostRingsMap::accessor ac;
        AddEdges(ids_neighbors, ac, t[0], t[1], t[2]);
        AddEdges(ids_neighbors, ac, t[1], t[2], t[0]);
        AddEdges(ids_neighbors, ac, t[2], t[0], t[1]);
    };

    tbb::parallel_for(size_t(0), triangles.size(), func);

    return ids_neighbors;
}