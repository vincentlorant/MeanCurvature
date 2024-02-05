#pragma once

#include <open3d/Open3D.h>

template <typename T> class MeanCurvatureSmooth {
  public:
    void operator()(std::shared_ptr<open3d::geometry::TriangleMesh> mesh,
                    int nitr = 50, float scale_factor = 0.035f);
};