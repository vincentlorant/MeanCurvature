#pragma once

#include <array>

using VertexId = int;

struct CudaEdge {
  VertexId end;
  std::array<VertexId, 2> adjs;
};

struct CudaRing {
  VertexId center;
  CudaEdge *edges;
  size_t edges_count = 0;
};