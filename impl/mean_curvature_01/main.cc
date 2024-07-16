#include <open3d/Open3D.h>

#include <argparse/argparse.hpp>

#include "common/helpers.h"
#include "mean_curvature.h"

int main(int argc, char **argv) {
  SetupLogger();

  auto parser = GetArgs(argc, argv);
  auto filename = parser->get<std::string>("--filename");
  int nitr = parser->get<int>("--nitr");
  float scale_factor = parser->get<float>("--scale_factor");

  auto mesh = GetMesh(filename, 0.02, true);

  TimeIt(MeanCurvature()(mesh, nitr, scale_factor));

  open3d::visualization::DrawGeometries({mesh}, "debug", 1280, 720, 100, 100,
                                        true);

  return 0;
}
