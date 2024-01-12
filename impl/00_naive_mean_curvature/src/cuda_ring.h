#pragma once

#include <Eigen/Dense>
#include <cuda_runtime.h>

#include "mean_curvature_types.h"

struct CudaEdge {
    VertexId end;
    VertexId adjs[2];
};

class CudaRing {
    VertexId center_;
    CudaEdge edges_[NUM_EDGES];
    int edges_count_ = 0;

  private:
    enum class AngleType { Obtuse, Acute };

    struct CurvatureData {
        double area;
        double cotan;
        AngleType angle_type;
    };

    struct CurvatureTerm {
        double k = 0.0;
        double w = 0.0;
        __device__ CurvatureTerm() = default;
        __device__ CurvatureTerm(double kv, double wv) : k(kv), w(wv) {}
        __device__ void operator+=(const CurvatureTerm &otr) {
            k += otr.k;
            w += otr.w;
        }
        __device__ double Compute() { return k / (4.0 * w); }
    };

  public:
    __host__ __device__ CudaRing();
    __host__ __device__ CudaRing(VertexId center) : center_(center) {}
    __inline__ __host__ __device__ int GetCenter() { return center_; }

    __inline__ __host__ __device__ void SetCenter(int center) {
        center_ = center;
    }

    __host__ __device__ void AddEdge(int end, int adj1, int adj2) {
        edges_[edges_count_++] = {end, adj1, adj2};
    }

    __device__ double GetSmallestEdgeLength(Eigen::Vector3d *positions);

    __device__ void UpdateCenterNormal(Eigen::Vector3d *positions,
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

    __device__ void ComputeCurvature(Eigen::Vector3d *positions,
                                     Eigen::Vector3d *normals) {
        CurvatureTerm ct;
        for (int i = 0; i < edges_count_; i++) {
            ct += ComputeEdgeCurvature(i, positions, normals);
        }
        double delta = ct.Compute();
        delta = max(-0.5, min(0.5, delta));
        positions[center_] -= normals[center_] * delta;
    }

  private:
    __device__ CurvatureTerm ComputeEdgeCurvature(int edge_idx,
                                                  Eigen::Vector3d *positions,
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

    __device__ CurvatureData GetCurvatureData(VertexId pc, VertexId pl,
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
        } else if (pcpl_n.dot(prpl_n) < epsilon ||
                   pcpr_n.dot(plpr_n) < epsilon) {
            w = plpc.cross(prpc).norm() * 0.125;
        } else {
            w = plpc.dot(plpc) * cotan * 0.125;
            atype = AngleType::Acute;
        }

        return {w, cotan, atype};
    }
};
