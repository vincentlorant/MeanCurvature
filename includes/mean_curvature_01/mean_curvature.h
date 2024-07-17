#include "common/host_ring.h"
#include "common/types.h"

class MeanCurvature {
  using MeshPtr = std::shared_ptr<open3d::geometry::TriangleMesh>;

 public:
  void operator()(MeshPtr mesh, int nitr = 50, double scale_factor = 0.2);

 private:
  auto CopyMeshDataToDevice(
    MeshPtr mesh) -> std::tuple<Eigen::Vector3d *, Eigen::Vector3d *, size_t>;
};
