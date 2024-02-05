#include <argparse/argparse.hpp>
#include <open3d/Open3D.h>

#include "helpers.hpp"
#include "mean_curvature.h"

int main(int argc, char **argv) {
    // setup
    SetupLogger();
    auto parser = GetArgs(argc, argv);

    // run algorithm
    auto mesh = GetMesh(parser->get<std::string>("--filename"), 0.01, true);
    TimeIt(MeanCurvatureSmooth<float>()(mesh, parser->get<int>("--nitr"),
                                        parser->get<float>("--scale_factor")));
    // visualize
    open3d::visualization::DrawGeometries({mesh});

    return EXIT_SUCCESS;
}