#include <open3d/Open3D.h>

#include "helpers.hpp"
#include "mean_curvature.h"

int main(int argv, char **argc) {
    SetupLogger();
    // inputs
    int nitr = argv > 1 ? std::stoi(argc[1]) : 50;
    float scale_factor = argv > 2 ? std::stof(argc[2]) : 0.035;
    auto filename = argv > 3 ? argc[3] : "../../assets/se_sub.ply";
    // run algorithm
    auto mesh = GetMesh(filename, 0.01, true);
    TimeIt(MeanCurvatureSmooth<float>()(mesh, nitr, scale_factor));
    // vizualize
    open3d::visualization::DrawGeometries({mesh});

    return EXIT_SUCCESS;
}