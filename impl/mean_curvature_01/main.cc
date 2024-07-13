#include <glog/logging.h>
#include <open3d/Open3D.h>

#include <argparse/argparse.hpp>

#include "common/helpers.h"
#include "mean_curvature.h"

int main(int argc, char **argv) {
  SetupLogger();
  auto parser = GetArgs(argc, argv);

  auto mesh = GetMesh(parser->get<std::string>("--filename"), 0.01, true);
  LOG(INFO) << mesh->vertices_.size();

  TimeIt(MeanCurvature()(mesh, parser->get<int>("--nitr")));

  open3d::visualization::DrawGeometries({mesh}, "debug", 1280, 720, 100, 100,
                                        true);

  return 0;
}
