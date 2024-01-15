#include "cuda_ring.h"
#include "mean_curvature_types.h"

#include <assert.h>
#include <float.h>

void CudaRing::CurvatureTerm::operator+=(const CurvatureTerm &otr) {
    k += otr.k;
    w += otr.w;
}

double CudaRing::CurvatureTerm::NormalizeCurvature() { return k / (4.0 * w); }

CudaRing::CudaRing(VertexId center, CudaEdge *edges, int edges_count)
    : center_(center), edges_(edges), edges_count_(edges_count) {}

CudaRing::~CudaRing() { /*cudaFree(edges_);*/ }

void CudaRing::SetCenter(int center) { center_ = center; }

void CudaRing::AddEdge(int end, int adj1, int adj2) {
    edges_[edges_count_++] = {end, adj1, adj2};
}

int CudaRing::GetCenter() { return center_; }

double CudaRing::GetSmallestEdgeLength(Eigen::Vector3d *positions) {
    double smallest = DBL_MAX;
    for (int i = 0; i < edges_count_; i++) {
        const CudaEdge &e = edges_[i];
        double length = (positions[e.end] - positions[center_]).norm();
        if (length < smallest) {
            smallest = length;
        }
    }

    return smallest;
}

void CudaRing::UpdateCenterNormal(Eigen::Vector3d *positions,
                                  Eigen::Vector3d *normals) {
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();

    for (int i = 0; i < edges_count_; i++) {
        const CudaEdge &e = edges_[i];
        int ie = e.end;
        int ia = e.adjs[0];
        int ib = e.adjs[1];
        Eigen::Vector3d c = positions[ie] - positions[center_];
        Eigen::Vector3d a = positions[ia] - positions[center_];
        Eigen::Vector3d b = positions[ib] - positions[center_];
        sum += a.cross(c);
        sum += b.cross(c);
    }
    normals[center_] = sum.normalized();
}

void CudaRing::ComputeCurvature(Eigen::Vector3d *positions,
                                Eigen::Vector3d *normals) {
    CurvatureTerm ct;
    for (int i = 0; i < edges_count_; i++) {
        ct += ComputeEdgeCurvature(i, positions, normals);
    }
    double delta = ct.NormalizeCurvature();
    delta = max(-0.5, min(0.5, delta));
    positions[center_] -= normals[center_] * delta;
}

CudaRing::CurvatureTerm
CudaRing::ComputeEdgeCurvature(int edge_idx, Eigen::Vector3d *positions,
                               Eigen::Vector3d *normals) {
    assert(edge_idx < edges_count_);
    const CudaEdge &e = edges_[edge_idx];
    int ie = e.end;
    int ir = e.adjs[0];
    int il = e.adjs[1];

    CurvatureData dl = GetCurvatureData(center_, ie, ir, positions);
    CurvatureData dr = GetCurvatureData(center_, ie, il, positions);

    const auto pcpi = positions[center_] - positions[ie];
    double forshortening = pcpi.dot(normals[center_]);
    double wk = 0.0;

    if (dl.angle_type == AngleType::Acute &&
        dr.angle_type == AngleType::Acute) {
        wk = (dr.cotan + dl.cotan) * forshortening;
    } else {
        double k = 2 * forshortening / pcpi.dot(pcpi);
        wk += dl.angle_type == AngleType::Acute ? dl.cotan * forshortening
                                                : dl.area * k;
        wk += dr.angle_type == AngleType::Acute ? dr.cotan * forshortening
                                                : dr.area * k;
    }

    return {wk, dr.area + dl.area};
}

CudaRing::CurvatureData CudaRing::GetCurvatureData(VertexId pc, VertexId pl,
                                                   VertexId pr,
                                                   Eigen::Vector3d *positions) {
    auto pcpr = positions[pc] - positions[pr];
    auto plpr = positions[pl] - positions[pr];
    auto prpc = positions[pr] - positions[pc];
    auto plpc = positions[pl] - positions[pc];
    auto pcpl = positions[pc] - positions[pl];
    auto prpl = positions[pr] - positions[pl];

    auto prpc_n = prpc.normalized();
    auto plpc_n = plpc.normalized();
    auto pcpr_n = pcpr.normalized();
    auto plpr_n = plpr.normalized();
    auto pcpl_n = pcpl.normalized();
    auto prpl_n = prpl.normalized();

    double epsilon = 0.0001;
    double w = 0.;
    AngleType atype = AngleType::Obtuse;
    double cotan = pcpr.dot(plpr) / pcpr.cross(plpr).norm();

    if (plpc_n.dot(prpc_n) < epsilon) {
        w = plpc.cross(prpc).norm() * 0.25;
    } else if (pcpl_n.dot(prpl_n) < epsilon || pcpr_n.dot(plpr_n) < epsilon) {
        w = plpc.cross(prpc).norm() * 0.125;
    } else {
        w = plpc.dot(plpc) * cotan * 0.125;
        atype = AngleType::Acute;
    }

    return {w, cotan, atype};
}