#include "cuda_ring.h"
#include "mean_curvature_types.h"

double CudaRing::GetSmallestEdgeLength(Eigen::Vector3d *positions) {
    double smallest = 1e20;
    for (int i = 0; i < edges_count_; i++) {
        const CudaEdge &e = edges_[i];
        double length = (positions[e.end] - positions[center_]).norm();
        if (length < smallest) {
            smallest = length;
        }
    }

    return smallest;
}