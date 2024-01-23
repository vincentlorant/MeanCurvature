#include "cuda_ring.h"
#include "mean_curvature_types.h"

#include <assert.h>
#include <float.h>

template <typename T>
void CudaRing<T>::CurvatureTerm::operator+=(const CurvatureTerm &otr) {
    k += otr.k;
    w += otr.w;
}

template <typename T> T CudaRing<T>::CurvatureTerm::NormalizeCurvature() {
    return k / (4.0 * w);
}

template <typename T>
CudaRing<T>::CudaRing(VertexId center, CudaEdge *edges, int edges_count)
    : center_(center), edges_(edges), edges_count_(edges_count) {}

template <typename T> CudaRing<T>::~CudaRing() {}

template <typename T> void CudaRing<T>::SetCenter(int center) {
    center_ = center;
}

template <typename T> void CudaRing<T>::AddEdge(int end, int adj1, int adj2) {
    edges_[edges_count_++] = {end, adj1, adj2};
}

template <typename T> int CudaRing<T>::GetCenter() { return center_; }

template <typename T> T CudaRing<T>::GetSmallestEdgeLength(Vector *positions) {
    T smallest = DBL_MAX;
    for (int i = 0; i < edges_count_; i++) {
        const CudaEdge &e = edges_[i];
        T length = (positions[e.end] - positions[center_]).norm();
        if (length < smallest) {
            smallest = length;
        }
    }

    return smallest;
}

template <typename T>
void CudaRing<T>::ComputeCurvature(Vector *positions, Vector *normals) {
    CurvatureTerm ct;
    for (int i = 0; i < edges_count_; i++) {
        ct += ComputeEdgeCurvature(i, positions, normals);
    }
    T delta = ct.NormalizeCurvature();
    delta = max(-0.5, min(0.5, delta));
    positions[center_] -= normals[center_] * delta;
}

template <typename T>
CudaRing<T>::CurvatureTerm CudaRing<T>::ComputeEdgeCurvature(int edge_idx,
                                                             Vector *positions,
                                                             Vector *normals) {
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
    T forshortening = pcpi.dot(normals[center_]);
    T wk = 0.0;

    if (dl.angle_type == AngleType::Acute &&
        dr.angle_type == AngleType::Acute) {
        wk = (dr.cotan + dl.cotan) * forshortening;
    } else {
        T k = 2 * forshortening / pcpi.dot(pcpi);
        wk += dl.angle_type == AngleType::Acute ? dl.cotan * forshortening
                                                : dl.area * k;
        wk += dr.angle_type == AngleType::Acute ? dr.cotan * forshortening
                                                : dr.area * k;
    }

    return {wk, dr.area + dl.area};
}

template <typename T>
CudaRing<T>::CurvatureData CudaRing<T>::GetCurvatureData(const Vector &pc,
                                                         const Vector &pe,
                                                         const Vector &px) {
    auto vcx = pc - px;
    auto vex = pe - px;
    auto vxc = px - pc;
    auto vec = pe - pc;
    auto vce = pc - pe;
    auto vxe = px - pe;

    T w = 0.;
    AngleType atype = AngleType::Obtuse;
    T cotan = vcx.dot(vex) / vcx.cross(vex).norm();

    if (vec.normalized().dot(vxc.normalized()) < 0) {
        w = vec.cross(vxc).norm() * 0.25;
    } else if (vce.normalized().dot(vxe.normalized()) < 0 ||
               vcx.normalized().dot(vex.normalized()) < 0) {
        w = vec.cross(vxc).norm() * 0.125;
    } else {
        w = vec.dot(vec) * cotan * 0.125;
        atype = AngleType::Acute;
    }

    return {w, cotan, atype};
}

// legal instantiations
template CudaRing<double>;
template CudaRing<float>;