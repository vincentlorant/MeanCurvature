#include <assert.h>
#include <device_functions.h>
#include <float.h>

#include <limits>

#include "cuda_ring.h"

template <typename T>
void CudaRing<T>::Curvature::operator+=(const Curvature &otr) {
  k = __fadd_rn(k, otr.k);
  w = __fadd_rn(w, otr.w);
}

template <typename T>
T CudaRing<T>::Curvature::NormalizeCurvature() {
  return __fdiv_rn(k, __fmul_rn(4.0, w));
}

template <typename T>
CudaRing<T>::CudaRing(VertexId center, CudaEdge *edges, int edges_count)
    : center_(center), edges_(edges), edges_count_(edges_count) {}

template <typename T>
CudaRing<T>::~CudaRing() {}

template <typename T>
int CudaRing<T>::GetCenter() const {
  return center_;
}

template <typename T>
T CudaRing<T>::GetSmallestEdgeLength(Vector *positions) const {
  T smallest = std::numeric_limits<T>::max();
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
void CudaRing<T>::ComputeCurvature(Vector *positions, Vector *normals) const {
  Curvature ct;
  Vector cn = normals[center_];
  for (int i = 0; i < edges_count_; i++) {
    ct += ComputeEdgeCurvature(i, positions, cn);
  }
  T delta = ct.NormalizeCurvature();
  delta = max(-0.5f, min(0.5f, delta));
  positions[center_] -= cn * delta;
}

template <typename T>
void CudaRing<T>::UpdateCenterNormal(Vector *positions, Vector *normals) const {
  Vector sum = Vector::Zero();

  for (int i = 0; i < edges_count_; i++) {
    const CudaEdge e = edges_[i];
    int ie = e.end;
    int ia = e.adjs[0];
    Vector pc = positions[center_];
    Vector pe = positions[ie] - pc;
    Vector pa = positions[ia] - pc;
    sum += pa.cross(pe);
  }
  normals[center_] = sum.normalized();
}

template <typename T>
CudaRing<T>::Curvature CudaRing<T>::ComputeEdgeCurvature(
    int edge_idx, Vector *positions, const Vector &center_normal) const {
  assert(edge_idx < edges_count_);
  const CudaEdge e = edges_[edge_idx];
  int ie = e.end;
  int ir = e.adjs[0];
  int il = e.adjs[1];

  auto pc = positions[center_];
  auto pe = positions[ie];

  CurvatureData dl = ComputeCurvatureWeight(pc, pe, positions[il]);
  CurvatureData dr = ComputeCurvatureWeight(pc, pe, positions[ir]);

  const auto pcpi = pc - pe;
  T forshortening = pcpi.dot(center_normal);
  T wk = 0.0;

  if (dl.angle_type == AngleType::Acute && dr.angle_type == AngleType::Acute) {
    wk = (dr.cotan + dl.cotan) * forshortening;
  } else {
    T k = __fdiv_rn(__fmul_rn(2, forshortening), pcpi.dot(pcpi));
    wk += dl.angle_type == AngleType::Acute ? __fmul_rn(dl.cotan, forshortening)
                                            : __fmul_rn(dl.area, k);
    wk += dr.angle_type == AngleType::Acute ? __fmul_rn(dr.cotan, forshortening)
                                            : __fmul_rn(dr.area, k);
  }

  return {wk, __fadd_rn(dr.area, dl.area)};
}

template <typename T>
CudaRing<T>::CurvatureData CudaRing<T>::ComputeCurvatureWeight(
    const Vector &pc, const Vector &pe, const Vector &px) const {
  auto vcx = pc - px;
  auto vex = pe - px;
  auto vxc = px - pc;
  auto vec = pe - pc;
  auto vce = pc - pe;
  auto vxe = px - pe;

  T w = 0.;
  AngleType atype = AngleType::Obtuse;
  T cotan = __fdiv_rn(vcx.dot(vex), vcx.cross(vex).norm());

  if (Normalized(vec).dot(Normalized(vxc)) < 0) {
    w = __fmul_rn(vec.cross(vxc).norm(), 0.25);
  } else if (Normalized(vce).dot(Normalized(vxe)) < 0 ||
             Normalized(vcx).dot(Normalized(vex)) < 0) {
    w = __fmul_rn(vec.cross(vxc).norm(), 0.125);
  } else {
    w = __fmul_rn(__fmul_rn(vec.dot(vec), cotan), 0.125);
    atype = AngleType::Acute;
  }

  return {w, cotan, atype};
}

template <typename T>
CudaRing<T>::Vector CudaRing<T>::Normalized(const Vector &v) const {
  T norm = Norm(v);

  return {__fdiv_rn(v(0), norm), __fdiv_rn(v(1), norm), __fdiv_rn(v(2), norm)};
}

template <typename T>
T CudaRing<T>::Norm(const Vector &v) const {
  T xx = __fmul_rn(v(0), v(0));
  T yy = __fmul_rn(v(1), v(1));
  T zz = __fmul_rn(v(2), v(2));

  return __fadd_rn(__fadd_rn(xx, yy), zz);
}

// legal instantiations
template class CudaRing<double>;
template class CudaRing<float>;