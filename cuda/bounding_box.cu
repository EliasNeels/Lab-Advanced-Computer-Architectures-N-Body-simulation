#include <cuda_runtime.h>
#include <cfloat>
#include "../include/quadtree.h"
#include "../include/body.h"
#include "../include/cuda_utils.h"
#include "../include/BoundingBox.h"

// =====================================================================
// Kernel 1: Bounding Box — Parallel min/max reduction
// =====================================================================

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
        atomicMin((int*)d_min_x, __float_as_int(s_min_x[0]));
        atomicMin((int*)d_min_y, __float_as_int(s_min_y[0]));
        // For max, we negate to use atomicMin trick (only works for positive floats)
        // Use atomicMax with int reinterpretation for positive floats
        atomicMax((int*)d_max_x, __float_as_int(s_max_x[0]));
        atomicMax((int*)d_max_y, __float_as_int(s_max_y[0]));
    }
}

// Host-side launcher
void launchBoundingBox(const Bodies& bodies, BoundingBox* d_bbox, cudaStream_t stream) {
    if (bodies.count == 0) return;

    // d_bbox layout: min_x, min_y, max_x, max_y (4 floats)
    // Use int representation for atomic compatibility
    int initInts[4];
    initInts[0] = __float_as_int(FLT_MAX);   // min_x initial
    initInts[1] = __float_as_int(FLT_MAX);   // min_y initial
    initInts[2] = __float_as_int(-FLT_MAX);  // max_x initial
    initInts[3] = __float_as_int(-FLT_MAX);  // max_y initial
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
