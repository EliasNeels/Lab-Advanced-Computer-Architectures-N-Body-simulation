#include <cuda_runtime.h>
#include <cfloat>
#include "../include/tree.h"
#include "../include/body.h"
#include "../include/cuda_utils.h"
#include "../include/BoundingBox.h"
#include "../include/kernels.cuh"

// =====================================================================
// Kernel 1: 3D Bounding Box — Parallel min/max reduction
//           Optimized: warp shuffle reduction before shared memory
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

// Warp-level min/max reduction (no shared memory needed within a warp)
__device__ __forceinline__ float warpReduceMin(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fminf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
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
    // Shared memory only needed for inter-warp reduction (1 entry per warp)
    const int warpsPerBlock = blockDim.x / 32;
    extern __shared__ float sdata[];
    float* s_min_x = sdata;
    float* s_min_y = sdata + warpsPerBlock;
    float* s_min_z = sdata + 2 * warpsPerBlock;
    float* s_max_x = sdata + 3 * warpsPerBlock;
    float* s_max_y = sdata + 4 * warpsPerBlock;
    float* s_max_z = sdata + 5 * warpsPerBlock;

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = tid & 31;
    int warpId = tid >> 5;

    float local_min_x = FLT_MAX, local_min_y = FLT_MAX, local_min_z = FLT_MAX;
    float local_max_x = -FLT_MAX, local_max_y = -FLT_MAX, local_max_z = -FLT_MAX;

    // Grid-stride loop for coalesced access
    for (int idx = i; idx < n; idx += blockDim.x * gridDim.x) {
        float px = __ldg(&pos_x[idx]);
        float py = __ldg(&pos_y[idx]);
        float pz = __ldg(&pos_z[idx]);
        local_min_x = fminf(local_min_x, px);
        local_min_y = fminf(local_min_y, py);
        local_min_z = fminf(local_min_z, pz);
        local_max_x = fmaxf(local_max_x, px);
        local_max_y = fmaxf(local_max_y, py);
        local_max_z = fmaxf(local_max_z, pz);
    }

    // Warp-level reduction (no shared memory, no __syncthreads)
    local_min_x = warpReduceMin(local_min_x);
    local_min_y = warpReduceMin(local_min_y);
    local_min_z = warpReduceMin(local_min_z);
    local_max_x = warpReduceMax(local_max_x);
    local_max_y = warpReduceMax(local_max_y);
    local_max_z = warpReduceMax(local_max_z);

    // Lane 0 of each warp writes to shared memory
    if (laneId == 0) {
        s_min_x[warpId] = local_min_x;
        s_min_y[warpId] = local_min_y;
        s_min_z[warpId] = local_min_z;
        s_max_x[warpId] = local_max_x;
        s_max_y[warpId] = local_max_y;
        s_max_z[warpId] = local_max_z;
    }
    __syncthreads();

    // First warp reduces across all warps
    if (warpId == 0) {
        local_min_x = (laneId < warpsPerBlock) ? s_min_x[laneId] : FLT_MAX;
        local_min_y = (laneId < warpsPerBlock) ? s_min_y[laneId] : FLT_MAX;
        local_min_z = (laneId < warpsPerBlock) ? s_min_z[laneId] : FLT_MAX;
        local_max_x = (laneId < warpsPerBlock) ? s_max_x[laneId] : -FLT_MAX;
        local_max_y = (laneId < warpsPerBlock) ? s_max_y[laneId] : -FLT_MAX;
        local_max_z = (laneId < warpsPerBlock) ? s_max_z[laneId] : -FLT_MAX;

        local_min_x = warpReduceMin(local_min_x);
        local_min_y = warpReduceMin(local_min_y);
        local_min_z = warpReduceMin(local_min_z);
        local_max_x = warpReduceMax(local_max_x);
        local_max_y = warpReduceMax(local_max_y);
        local_max_z = warpReduceMax(local_max_z);

        if (laneId == 0) {
            atomicMinFloat(d_min_x, local_min_x);
            atomicMinFloat(d_min_y, local_min_y);
            atomicMinFloat(d_min_z, local_min_z);
            atomicMaxFloat(d_max_x, local_max_x);
            atomicMaxFloat(d_max_y, local_max_y);
            atomicMaxFloat(d_max_z, local_max_z);
        }
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
    int warpsPerBlock = blockSize / 32;
    int sharedMem = 6 * warpsPerBlock * sizeof(float);  // Much less shared memory

    kernelBoundingBox<<<numBlocks, blockSize, sharedMem, stream>>>(
        bodies.pos_x, bodies.pos_y, bodies.pos_z, bodies.count,
        (float*)d_bbox + 0, (float*)d_bbox + 1, (float*)d_bbox + 2,
        (float*)d_bbox + 3, (float*)d_bbox + 4, (float*)d_bbox + 5
    );
}
