#pragma once

using VertexId = int;

struct CudaEdge {
  VertexId end;
  VertexId adjs[2];
};

struct CudaRing {
  VertexId center;
  CudaEdge *edges;
  size_t edges_count = 0;

  // CudaRing(VertexId center, CudaEdge *edges, int edges_count);
};