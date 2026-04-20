#include <cuda_runtime.h>
#include <cfloat>
#include "../include/quadtree.h"
#include "../include/body.h"
#include "../include/cuda_utils.h"
#include "../include/BoundingBox.h"
#include "../include/kernels.cuh"

// =====================================================================
// Kernel 1: Bounding Box — Parallel min/max reduction
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
                                   int n,
                                   float* __restrict__ d_min_x,
                                   float* __restrict__ d_min_y,
                                   float* __restrict__ d_max_x,
                                   float* __restrict__ d_max_y) {
    extern __shared__ float sdata[];
    // shared layout: [0..blockDim.x) = min_x, [blockDim.x..2*blockDim.x) = min_y, etc
    float* s_min_x = sdata;
    float* s_min_y = sdata + blockDim.x;
    float* s_max_x = sdata + 2 * blockDim.x;
    float* s_max_y = sdata + 3 * blockDim.x;

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float local_min_x = FLT_MAX, local_min_y = FLT_MAX;
    float local_max_x = -FLT_MAX, local_max_y = -FLT_MAX;

    // Grid-stride loop for large N
    for (int idx = i; idx < n; idx += blockDim.x * gridDim.x) {
        float px = pos_x[idx];
        float py = pos_y[idx];
        local_min_x = fminf(local_min_x, px);
        local_min_y = fminf(local_min_y, py);
        local_max_x = fmaxf(local_max_x, px);
        local_max_y = fmaxf(local_max_y, py);
    }

    s_min_x[tid] = local_min_x;
    s_min_y[tid] = local_min_y;
    s_max_x[tid] = local_max_x;
    s_max_y[tid] = local_max_y;
    __syncthreads();

    // Reduction within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_min_x[tid] = fminf(s_min_x[tid], s_min_x[tid + s]);
            s_min_y[tid] = fminf(s_min_y[tid], s_min_y[tid + s]);
            s_max_x[tid] = fmaxf(s_max_x[tid], s_max_x[tid + s]);
            s_max_y[tid] = fmaxf(s_max_y[tid], s_max_y[tid + s]);
        }
        __syncthreads();
    }

    // Atomic update global result
    if (tid == 0) {
        atomicMinFloat(d_min_x, s_min_x[0]);
        atomicMinFloat(d_min_y, s_min_y[0]);
        atomicMaxFloat(d_max_x, s_max_x[0]);
        atomicMaxFloat(d_max_y, s_max_y[0]);
    }
}

// Host-side launcher
void launchBoundingBox(const Bodies& bodies, BoundingBox* d_bbox, cudaStream_t stream) {
    if (bodies.count == 0) return;

    // d_bbox layout: min_x, min_y, max_x, max_y (4 floats)
    // Use int representation for atomic compatibility
    int initInts[4];
    float f1 = FLT_MAX;
    float f2 = FLT_MAX;
    float f3 = -FLT_MAX;
    float f4 = -FLT_MAX;
    initInts[0] = *reinterpret_cast<int*>(&f1);   // min_x initial
    initInts[1] = *reinterpret_cast<int*>(&f2);   // min_y initial
    initInts[2] = *reinterpret_cast<int*>(&f3);  // max_x initial
    initInts[3] = *reinterpret_cast<int*>(&f4);  // max_y initial
    CUDA_CHECK(cudaMemcpyAsync(d_bbox, initInts, 4 * sizeof(float),
                                cudaMemcpyHostToDevice, stream));

    int blockSize = 256;
    int numBlocks = min(256, (bodies.count + blockSize - 1) / blockSize);
    int sharedMem = 4 * blockSize * sizeof(float);

    kernelBoundingBox<<<numBlocks, blockSize, sharedMem, stream>>>(
        bodies.pos_x, bodies.pos_y, bodies.count,
        (float*)d_bbox + 0, (float*)d_bbox + 1,
        (float*)d_bbox + 2, (float*)d_bbox + 3
    );
}
