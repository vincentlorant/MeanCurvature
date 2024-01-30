#pragma once

#include <cuda_runtime.h>

#define cuCheck(ans)                                                           \
    { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
            exit(code);
    }
}

#define cuAsyncCheck()                                                         \
    cudaError_t cerror;                                                        \
    cudaCheckLast(cerror);

#define cuCheckLast(cerror)                                                    \
    cerror = cudaGetLastError();                                               \
    if (cerror != cudaSuccess) {                                               \
        printf("Error: %s\n", cudaGetErrorString(cerror));                     \
    }

