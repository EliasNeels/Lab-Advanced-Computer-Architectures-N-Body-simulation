#include <cuda_runtime.h>
#include <cfloat>
#include "../include/tree.h"
#include "../include/body.h"
#include "../include/cuda_utils.h"
#include "../include/BoundingBox.h"
#include "../include/kernels.cuh"

// =====================================================================
// Kernel 1: 3D Bounding Box — Parallel min/max reduction
// =====================================================================

__device__ __forceinline__ float atomicMinFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) <= val) break;
        old = atomicCAS(address_as_int, assumed, __float_as_int(val));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ __forceinline__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= val) break;
        old = atomicCAS(address_as_int, assumed, __float_as_int(val));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void kernelBoundingBox(const float* __restrict__ pos_x,
                                   const float* __restrict__ pos_y,
                                   const float* __restrict__ pos_z,
                                   int n,
                                   float* __restrict__ d_min_x,
                                   float* __restrict__ d_min_y,
                                   float* __restrict__ d_min_z,
                                   float* __restrict__ d_max_x,
                                   float* __restrict__ d_max_y,
                                   float* __restrict__ d_max_z) {
    extern __shared__ float sdata[];
    float* s_min_x = sdata;
    float* s_min_y = sdata + blockDim.x;
    float* s_min_z = sdata + 2 * blockDim.x;
    float* s_max_x = sdata + 3 * blockDim.x;
    float* s_max_y = sdata + 4 * blockDim.x;
    float* s_max_z = sdata + 5 * blockDim.x;

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float local_min_x = FLT_MAX, local_min_y = FLT_MAX, local_min_z = FLT_MAX;
    float local_max_x = -FLT_MAX, local_max_y = -FLT_MAX, local_max_z = -FLT_MAX;

    for (int idx = i; idx < n; idx += blockDim.x * gridDim.x) {
        float px = pos_x[idx];
        float py = pos_y[idx];
        float pz = pos_z[idx];
        local_min_x = fminf(local_min_x, px);
        local_min_y = fminf(local_min_y, py);
        local_min_z = fminf(local_min_z, pz);
        local_max_x = fmaxf(local_max_x, px);
        local_max_y = fmaxf(local_max_y, py);
        local_max_z = fmaxf(local_max_z, pz);
    }

    s_min_x[tid] = local_min_x;
    s_min_y[tid] = local_min_y;
    s_min_z[tid] = local_min_z;
    s_max_x[tid] = local_max_x;
    s_max_y[tid] = local_max_y;
    s_max_z[tid] = local_max_z;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_min_x[tid] = fminf(s_min_x[tid], s_min_x[tid + s]);
            s_min_y[tid] = fminf(s_min_y[tid], s_min_y[tid + s]);
            s_min_z[tid] = fminf(s_min_z[tid], s_min_z[tid + s]);
            s_max_x[tid] = fmaxf(s_max_x[tid], s_max_x[tid + s]);
            s_max_y[tid] = fmaxf(s_max_y[tid], s_max_y[tid + s]);
            s_max_z[tid] = fmaxf(s_max_z[tid], s_max_z[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMinFloat(d_min_x, s_min_x[0]);
        atomicMinFloat(d_min_y, s_min_y[0]);
        atomicMinFloat(d_min_z, s_min_z[0]);
        atomicMaxFloat(d_max_x, s_max_x[0]);
        atomicMaxFloat(d_max_y, s_max_y[0]);
        atomicMaxFloat(d_max_z, s_max_z[0]);
    }
}

void launchBoundingBox(const Bodies& bodies, BoundingBox* d_bbox, cudaStream_t stream) {
    if (bodies.count == 0) return;

    // Initialize 6 floats: min_x, min_y, min_z, max_x, max_y, max_z
    int initInts[6];
    float fmax = FLT_MAX, fmin = -FLT_MAX;
    initInts[0] = *reinterpret_cast<int*>(&fmax);  // min_x
    initInts[1] = *reinterpret_cast<int*>(&fmax);  // min_y
    initInts[2] = *reinterpret_cast<int*>(&fmax);  // min_z
    initInts[3] = *reinterpret_cast<int*>(&fmin);  // max_x
    initInts[4] = *reinterpret_cast<int*>(&fmin);  // max_y
    initInts[5] = *reinterpret_cast<int*>(&fmin);  // max_z
    CUDA_CHECK(cudaMemcpyAsync(d_bbox, initInts, 6 * sizeof(float),
                                cudaMemcpyHostToDevice, stream));

    int blockSize = 256;
    int numBlocks = min(256, (bodies.count + blockSize - 1) / blockSize);
    int sharedMem = 6 * blockSize * sizeof(float);

    kernelBoundingBox<<<numBlocks, blockSize, sharedMem, stream>>>(
        bodies.pos_x, bodies.pos_y, bodies.pos_z, bodies.count,
        (float*)d_bbox + 0, (float*)d_bbox + 1, (float*)d_bbox + 2,
        (float*)d_bbox + 3, (float*)d_bbox + 4, (float*)d_bbox + 5
    );
}
