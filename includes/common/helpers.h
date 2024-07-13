#pragma once

#include <glog/logging.h>
#include <oneapi/tbb.h>
#include <open3d/Open3D.h>

#include <argparse/argparse.hpp>
#include <chrono>
#include <filesystem>
#include <memory>
#include <string>

#define time_duration std::chrono::duration_cast<std::chrono::milliseconds>
#define TimeIt(func)                                   \
  auto t1 = std::chrono::high_resolution_clock::now(); \
  func;                                                \
  auto t2 = std::chrono::high_resolution_clock::now(); \
  LOG(INFO) << time_duration(t2 - t1).count() << "ms";

void SetupLogger();

std::shared_ptr<argparse::ArgumentParser> GetArgs(int argc, char **argv);

void ApplyNoise(std::shared_ptr<open3d::geometry::TriangleMesh> pmesh,
                float noise);

std::shared_ptr<open3d::geometry::TriangleMesh> GetMesh(
    const std::string &filename, float noise = 0.01, bool debug = true);