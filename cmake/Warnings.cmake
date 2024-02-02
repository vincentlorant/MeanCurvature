# attention! suppress annoying warnings from eigen tbb & open3d
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; 
    -Xcudafe "--diag_suppress=20015"
    -Xcudafe "--diag_suppress=20013"
    -Xcudafe "--diag_suppress=20037"
    -Xcudafe "--diag_suppress=1675"
    -Xcudafe "--diag_suppress=611"
    -Xcudafe "--diag_suppress=550"
    -Xcudafe "--diag_suppress=186"
    -Xcudafe "--diag_suppress=28"
)