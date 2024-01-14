#include "host_ring.h"

#include <oneapi/tbb.h>
#include <open3d/Open3D.h>
#include <unordered_map>
#include <vector>

#include "cuda_ring.h"
#include "mean_curvature_types.h"

void HostRing::EdgeData::AddAdj(const VertexId &id) {
    uint8_t array_num = sizeof(adjs) / sizeof(adjs[0]);
    if (count < array_num) {
        adjs[count++] = id;
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

CudaRing HostRing::ToCudaRing() const {
    if (edges_.size() > MAX_NUM_EDGES_PER_RING) {
        LOG(FATAL) << "edges size:" << edges_.size()
                   << " should not be greater than MAX_NUM_EDGES_PER_RING:"
                   << MAX_NUM_EDGES_PER_RING;
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

const std::unordered_map<VertexId, HostRing::EdgeData> &
HostRing::GetEdges() const {
    return edges_;
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