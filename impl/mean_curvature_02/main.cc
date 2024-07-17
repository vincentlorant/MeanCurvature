#include <open3d/Open3D.h>

#include <argparse/argparse.hpp>

#include "common/helpers.h"
#include "mean_curvature.h"

int main(int argc, char **argv) {
  // setup
  SetupLogger();
  auto parser = GetArgs(argc, argv);
  // get params
  auto filename = parser->get<std::string>("--filename");
  int nitr = parser->get<int>("--nitr");
  float scale_factor = parser->get<float>("--scale_factor");
  // run algorithm
  auto mesh = GetMesh(filename, 0.01, true);
  TimeIt(MeanCurvature<float>()(mesh, nitr, scale_factor));
  // visualize
  open3d::visualization::DrawGeometries({mesh});

  return EXIT_SUCCESS;
}