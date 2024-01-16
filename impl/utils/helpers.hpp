#pragma once

#include <chrono>
#include <filesystem>
#include <glog/logging.h>
#include <memory>
#include <open3d/Open3D.h>
#include <string>

#include <cuda_runtime.h>

#define time_duration std::chrono::duration_cast<std::chrono::milliseconds>
#define TimeIt(func)                                                           \
    auto t1 = std::chrono::high_resolution_clock::now();                       \
    func;                                                                      \
    auto t2 = std::chrono::high_resolution_clock::now();                       \
    LOG(INFO) << time_duration(t2 - t1).count() << "ms";

#define TimeCudaEvent(func, itr)                                               \
    cudaEvent_t start, stop;                                                   \
    cudaEventCreate(&start);                                                   \
    cudaEventCreate(&stop);                                                    \
    cudaEventRecord(start);                                                    \
    func;                                                                      \
    cudaEventRecord(stop);                                                     \
    cudaEventSynchronize(stop);                                                \
    float time;                                                                \
    cudaEventElapsedTime(&time, start, stop);                                  \
    cudaEventDestroy(start);                                                   \
    cudaEventDestroy(stop);                                                    \
    LOG(INFO) << "kernel avg time:" << (time / itr) << "ms";

void SetupLogger() {
    google::InitGoogleLogging("mean_curvature");
    FLAGS_alsologtostderr = 1;
    FLAGS_colorlogtostderr = 1;
}

std::shared_ptr<open3d::geometry::TriangleMesh>
GetMesh(const std::string &filename, float noise_scale = 0.01,
        bool debug = true, Eigen::Vector3d color = {0.3, 0.5, 0.4}) {
    if (std::filesystem::exists(filename) == false) {
        LOG(FATAL) << "Invalid mesh file:" << filename;
    }
    open3d::geometry::TriangleMesh mesh;
    open3d::io::ReadTriangleMesh(filename, mesh);
    auto pmesh = std::make_shared<open3d::geometry::TriangleMesh>(mesh);
    pmesh->ComputeVertexNormals();

    for (int i = 0; i < pmesh->vertices_.size(); i++) {
        float rnd = (float)std::rand() / RAND_MAX;
        pmesh->vertices_[i] += pmesh->vertex_normals_[i] * rnd * noise_scale;
    }
    pmesh->ComputeVertexNormals();

    if (debug) {
        pmesh->PaintUniformColor(color);
        open3d::visualization::DrawGeometries({pmesh}, "debug", 1280, 720, 100,
                                              100);
    }

    return pmesh;
}