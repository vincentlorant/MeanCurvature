#include <glog/logging.h>
#include <oneapi/tbb.h>
#include <open3d/Open3D.h>

#include <argparse/argparse.hpp>
#include <chrono>
#include <filesystem>
#include <memory>
#include <string>

void SetupLogger() {
  google::InitGoogleLogging("mean_curvature");
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;
}

std::shared_ptr<argparse::ArgumentParser> GetArgs(int argc, char **argv) {
  auto parser = std::make_shared<argparse::ArgumentParser>("mean-curvature");

  parser->add_argument("--nitr")
      .default_value(50)
      .help("Number of mean-curvature iterations")
      .scan<'i', int>();

  parser->add_argument("--scale_factor")
      .default_value(0.2f)
      .help(
          "Scalar multiplier to scale mesh before processing. Scaling down "
          "reduce the needed number of iterations")
      .scan<'g', float>();
  ;

  parser->add_argument("--filename")
      .default_value(std::string{"../../assets/se_sub.ply"})
      .help("Open3d compatible mesh path");

  try {
    parser->parse_args(argc, argv);
  } catch (const std::exception &err) {
    LOG(ERROR) << err.what();
    std::exit(1);
  }

  return parser;
}

void ApplyNoise(std::shared_ptr<open3d::geometry::TriangleMesh> pmesh,
                float noise) {
  pmesh->ComputeVertexNormals();
  for (int i = 0; i < pmesh->vertices_.size(); i++) {
    float rnd = (float)std::rand() / RAND_MAX;
    pmesh->vertices_[i] += pmesh->vertex_normals_[i] * rnd * noise;
  }
}

std::shared_ptr<open3d::geometry::TriangleMesh> GetMesh(
    const std::string &filename, float noise = 0.01, bool debug = true) {
  if (std::filesystem::exists(filename) == false) {
    LOG(FATAL) << "Invalid mesh file:" << filename;
  }
  open3d::geometry::TriangleMesh mesh;
  open3d::io::ReadTriangleMesh(filename, mesh);
  auto pmesh = std::make_shared<open3d::geometry::TriangleMesh>(mesh);
  if (debug) {
    pmesh->ComputeVertexNormals();
    open3d::visualization::DrawGeometries({pmesh});
  }
  ApplyNoise(pmesh, noise);
  pmesh->ComputeVertexNormals();

  if (debug) {
    open3d::visualization::DrawGeometries({pmesh});
  }

  return pmesh;
}