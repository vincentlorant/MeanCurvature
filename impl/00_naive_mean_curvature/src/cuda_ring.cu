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

CudaRing::~CudaRing() { /*cudaFree(edges_);*/
}

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

    auto pc = positions[center_];
    auto pe = positions[ie];

    CurvatureData dl = GetCurvatureData(pc, pe, positions[il]);
    CurvatureData dr = GetCurvatureData(pc, pe, positions[ir]);

    const auto pcpi = pc - pe;
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

CudaRing::CurvatureData CudaRing::GetCurvatureData(const Eigen::Vector3d &pc,
                                                   const Eigen::Vector3d &pe,
                                                   const Eigen::Vector3d &px) {
    auto vcx = pc - px;
    auto vex = pe - px;
    auto vxc = px - pc;
    auto vec = pe - pc;
    auto vce = pc - pe;
    auto vxe = px - pe;

    double w = 0.;
    double cotan = 0;
    AngleType atype = AngleType::Obtuse;

    if (vec.normalized().dot(vxc.normalized()) < 0) {
        w = vec.cross(vxc).norm() * 0.25;
        vcx.dot(vex) / vcx.cross(vex).norm();
    } else if (vce.normalized().dot(vxe.normalized()) < 0 ||
               vcx.normalized().dot(vex.normalized()) < 0) {
        w = vec.cross(vxc).norm() * 0.125;
        vcx.dot(vex) / vcx.cross(vex).norm();
    } else {
        w = vec.dot(vec) * cotan * 0.125;
        atype = AngleType::Acute;
    }

    return {w, cotan, atype};
}