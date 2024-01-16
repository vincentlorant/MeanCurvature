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

CudaRing::CurvatureData CudaRing::GetCurvatureData(VertexId ie, VertexId ia,
                                                   VertexId ib,
                                                   Eigen::Vector3d *positions) {
    auto ieib = positions[ie] - positions[ib];
    auto iaib = positions[ia] - positions[ib];
    auto ibie = positions[ib] - positions[ie];
    auto iaie = positions[ia] - positions[ie];
    auto ieia = positions[ie] - positions[ia];
    auto ibia = positions[ib] - positions[ia];

    auto ibie_n = ibie.normalized();
    auto iaie_n = iaie.normalized();
    auto ieib_n = ieib.normalized();
    auto iaib_n = iaib.normalized();
    auto ieia_n = ieia.normalized();
    auto ibia_n = ibia.normalized();

    double epsilon = 0.0001;
    double w = 0.;
    AngleType atype = AngleType::Obtuse;
    double cotan = ieib.dot(iaib) / ieib.cross(iaib).norm();

    if (iaie_n.dot(ibie_n) < epsilon) {
        w = iaie.cross(ibie).norm() * 0.25;
    } else if (ieia_n.dot(ibia_n) < epsilon || ieib_n.dot(iaib_n) < epsilon) {
        w = iaie.cross(ibie).norm() * 0.125;
    } else {
        w = iaie.dot(iaie) * cotan * 0.125;
        atype = AngleType::Acute;
    }

    return {w, cotan, atype};
}