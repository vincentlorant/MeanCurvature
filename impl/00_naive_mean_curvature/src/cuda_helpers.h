#pragma once

#include <cuda_runtime.h>

#define cudaCheck(ans)                                                         \
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

#define cudaAsyncCheck()                                                       \
    cudaError_t cerror;                                                        \
    cudaCheckLast(cerror);

#define cudaCheckLast(cerror)                                                  \
    cerror = cudaGetLastError();                                               \
    if (cerror != cudaSuccess) {                                               \
        printf("Error: %s\n", cudaGetErrorString(cerror));                     \
    }
